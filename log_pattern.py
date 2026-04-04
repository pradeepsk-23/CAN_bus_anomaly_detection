import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EdgeLogEmbedder:
    """
    Handles lightweight Transformer embeddings optimized for CPU-bound edge targets.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_length: int = 128):
        # Using MiniLM or DistilBERT as they are heavily optimized for edge inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logging.info(f"Initialized EdgeLogEmbedder on {self.device} with {model_name}")

    def sliding_window_tokenize(self, text: str) -> list:
        """
        Implements sliding window tokenization for long-form log sequences.
        """
        tokens = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=True
        )['input_ids'][0]
        
        # If sequence is short, return as is
        if len(tokens) <= self.max_length:
            return [tokens]
            
        # Sliding window logic
        stride = self.max_length // 2
        windows = []
        for i in range(0, len(tokens), stride):
            window = tokens[i : i + self.max_length]
            if len(window) < self.max_length:
                # Pad the last window
                padding = torch.zeros(self.max_length - len(window), dtype=torch.long)
                window = torch.cat([window, padding])
            windows.append(window)
        return windows

    def get_embeddings(self, log_sequences: list[str]) -> np.ndarray:
        """
        Generates embeddings for a batch of logs.
        """
        embeddings = []
        with torch.no_grad():
            for text in log_sequences:
                # In a real edge scenario, we average the sliding windows
                windows = self.sliding_window_tokenize(text)
                window_embeddings = []
                for window in windows:
                    # Add batch dimension
                    input_ids = window.unsqueeze(0).to(self.device)
                    # Create attention mask
                    attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    # Use mean pooling for the sequence representation
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    window_embeddings.append(emb)
                
                # Average the embeddings of all windows for a single log trace
                embeddings.append(np.mean(window_embeddings, axis=0))
                
        return np.array(embeddings)


class RegressionClusterer:
    """
    Handles Unsupervised clustering to detect crash precursors.
    """
    def __init__(self, pca_components: int = 16, eps: float = 0.5, min_samples: int = 5):
        self.pca = PCA(n_components=pca_components)
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        logging.info("Initialized RegressionClusterer with PCA + DBSCAN")

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduces dimensionality to simulate CPU-bound constraints, then clusters.
        """
        logging.info("Applying PCA for dimensionality reduction...")
        reduced_embeddings = self.pca.fit_transform(embeddings)
        
        logging.info("Clustering with DBSCAN...")
        labels = self.dbscan.fit_predict(reduced_embeddings)
        return labels


class LLMTriageSummarizer:
    """
    Integrates with Cloud LLM via LangChain for developer triage.
    """
    def __init__(self, api_key: str):
        # Using Gemini as the cloud LLM, as mentioned in JD
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        self.prompt_template = PromptTemplate(
            input_variables=["log_samples"],
            template="""
            You are an expert Automotive Edge Software Engineer. Review the following cluster of vehicle system logs.
            These logs have been identified as anomalous software regressions or potential crash precursors.
            
            Logs:
            {log_samples}
            
            Provide a concise, human-readable summary of the potential root cause (e.g., memory leak, race condition, network timeout). 
            Format as a quick triage report.
            """
        )
        self.chain = self.prompt_template | self.llm

    def summarize_cluster(self, log_samples: list[str]) -> str:
        logs_text = "\n".join(log_samples)
        response = self.chain.invoke({"log_samples": logs_text})
        return response.content

# ==========================================
# Example Pipeline Execution
# ==========================================
if __name__ == "__main__":
    # 1. Simulate data ingestion (normally you'd load BGL dataset via pandas here)
    sample_dmesg_logs = [
        "eth0: link up, 1000Mbps, full-duplex",
        "systemd: Started vehicle-telemetry.service",
        "kernel: nvme0n1: p1 p2 p3",
        # Anomalous/Regression logs simulating a crash precursor
        "kernel: [112.33] memory allocation failed: out of memory",
        "kernel: [112.34] kswapd0: page allocation failure",
        "app_manager: watchdog timeout on thread 4, restarting service"
    ]
    
    # 2. Embed the logs
    embedder = EdgeLogEmbedder()
    embeddings = embedder.get_embeddings(sample_dmesg_logs)
    
    # 3. Cluster the logs
    clusterer = RegressionClusterer(pca_components=3, eps=0.3, min_samples=2)
    labels = clusterer.fit_predict(embeddings)
    
    # 4. Triage Anomalies
    # DBSCAN labels -1 as noise/anomaly. 
    anomalous_logs = [log for log, label in zip(sample_dmesg_logs, labels) if label == -1]
    
    if anomalous_logs:
        logging.info(f"Detected {len(anomalous_logs)} anomalous logs. Sending to LLM for triage...")
        # summarizer = LLMTriageSummarizer(api_key="YOUR_API_KEY")
        # report = summarizer.summarize_cluster(anomalous_logs)
        # print("\n--- Triage Report ---\n", report)