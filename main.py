import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.svm import SVC
def load_data(file_path):
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            for event in data["content"]:
                events.append({
                    "id": data["id"],
                    "sentence": event["sentence"],
                    "tokens": event["tokens"],
                    "eventype": event["eventype"],
                    "eventype_id": event["eventype_id"],
                    "trigger": event["trigger"],
                    "position": event["position"]
                })
    return pd.DataFrame(events)

class BERTDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def fine_tune_bert(X_train, y_train, X_test, y_test, model_name='bert-base-uncased', epochs=3, batch_size=8,
                   max_len=128):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = BERTDataset(X_train, y_train, tokenizer, max_len)
    test_dataset = BERTDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y_train)))
    model = model.cuda() if torch.cuda.is_available() else model

    optimizer = AdamW(model.parameters(), lr=2e-5)


    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
            attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
            labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
            attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
            labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    return accuracy, report

def create_pairs(X, threshold=0.7):
    positive_pairs = []
    negative_pairs = []

    similarity_matrix = cosine_similarity(X)

    # Create positive and negative pairs based on similarity
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if similarity_matrix[i, j] > threshold:  # Positive pair
                positive_pairs.append((i, j))
            else:  # Negative pair
                negative_pairs.append((i, j))

    return positive_pairs, negative_pairs

def clustering_and_classification(df, X_train, y_train, X_test, y_test):
    # Apply KMeans clustering to the training data, set number of clusters to 9
    kmeans = KMeans(n_clusters=9, random_state=42)  # Clustering into 9 categories
    df['cluster'] = kmeans.fit_predict(X_train)

    # Train a Support Vector Machine classifier
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = svm_classifier.predict(X_test)

    # Return the accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report

def main():
    # Load the data
    df = load_data('data.jsonl')
    print(f"Loaded data: {df.head()}")

    # Encode the event types
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['eventype'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['sentence'], y, test_size=0.3, random_state=42)

    # Fine-tune BERT model
    accuracy, report = fine_tune_bert(X_train, y_train, X_test, y_test)

    # Output results
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Perform clustering and classification
    clustering_accuracy, clustering_report = clustering_and_classification(df, X_train, y_train, X_test, y_test)
    print(f"Clustering Accuracy: {clustering_accuracy}")
    print(f"Clustering Classification Report:\n{clustering_report}")

if __name__ == "__main__":
    main()

