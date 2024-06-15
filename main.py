
import collections
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import tqdm
import transformers
from torch.utils.data import DataLoader
import pandas as pd

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data,validation_data,test_data = load_dataset("dair-ai/emotion", split=["train","validation","test"], trust_remote_code=True)

print(train_data, validation_data, test_data)

pd.DataFrame({
    "text":train_data['text'][:10],
    "label":train_data['label'][:10],
})

values_train,counts_train=np.unique(train_data['label'], return_counts=True)
values_validation,counts_validation=np.unique(validation_data['label'], return_counts=True)
values_test,counts_test=np.unique(test_data['label'], return_counts=True)

x_labels= ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1).title.set_text("Training set")
plt.bar(values_train, counts_train)
plt.xticks(values_train, x_labels)
plt.tight_layout()

plt.subplot(1,3,2).title.set_text("Validation set")
plt.bar(values_validation, counts_validation)
plt.xticks(values_validation, x_labels)
plt.tight_layout()

plt.subplot(1, 3, 3).title.set_text("Testing set")
plt.bar(values_test, counts_test)
plt.xticks(values_test, x_labels)
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)

plt.show()

transformer_name ="bert-base-cased"

tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name,return_attention_mask=False, return_token_type_ids=False)

tokenizer.tokenize("hello world!")

tokenizer.encode("hello world!")

tokenizer.convert_ids_to_tokens(tokenizer.encode("hello world"))

tokenizer("hello world!")

def tokenize(data_point):
    ids = tokenizer(data_point["text"])["input_ids"]
    return {"ids": ids}

train_data=train_data.map(tokenize)
validation_data=validation_data.map(tokenize)
test_data=test_data.map(tokenize)

pd.DataFrame({
    "Text":train_data["text"][:10],
    "Tokens":train_data["ids"][:10],
    "Length of tokens": [len(train_data["ids"][i]) for i in range(10)]
})

pad_index = tokenizer.pad_token_id

def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

train_data = train_data.with_format(type="torch", columns=["ids", "label"])
validation_data = validation_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])

batch_size = 8

train_data_loader=DataLoader(dataset=train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
validation_data_loader=DataLoader(dataset=validation_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader=DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

transformer = transformers.AutoModel.from_pretrained(transformer_name)

class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids):
        output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        attention = output.attentions[-1]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        return prediction


## source: https://github.com/bentrevett/pytorch-sentiment-analysis

output_dim = 6 #we want to get the size of the output layer, which in this case should be the size of the number of classes we have
freeze = False

model = Transformer(transformer, output_dim, freeze)


lr = 1e-5

optimizer = optim.Adam(model.parameters(), lr=lr)

loss_fn = nn.CrossEntropyLoss()

def train(data_loader, model, loss_fn, optimizer, device):
    model.train() #set the model in training mode
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = loss_fn(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


  ## source: https://github.com/bentrevett/pytorch-sentiment-analysis

def evaluate(data_loader, model, loss_fn, device):
    model.eval() #set the model in evaluation mode
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = loss_fn(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


  ## source: https://github.com/bentrevett/pytorch-sentiment-analysis

def get_accuracy(prediction, label):
    predicted_classes = prediction.argmax(dim=-1)
    correct = torch.where(predicted_classes == label, 1, 0).sum()
    accuracy = correct / len(label)
    return accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
loss_fn = loss_fn.to(device)

n_epochs = 3
best_validation_loss = float("inf")

train_losses=[]
train_accs=[]
validation_losses=[]
validation_accs=[]

for epoch in range(n_epochs):
    train_loss, train_acc = train(
        train_data_loader, model, loss_fn, optimizer, device
    )
    validation_loss, validation_acc = evaluate(validation_data_loader, model, loss_fn, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    validation_losses.append(validation_loss)
    validation_accs.append(validation_acc)
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model.state_dict(), transformer_name+".pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"validation_loss: {validation_loss:.3f}, validation_acc: {validation_acc:.3f}")

def plot_losses():
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_losses, label="train loss")
    ax.plot(validation_losses, label="valid loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()

plot_losses()

def plot_accs():
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_accs, label="train accuracy")
    ax.plot(validation_accs, label="valid accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()

plot_accs()

model.load_state_dict(torch.load(transformer_name+".pt"))

test_loss, test_acc = evaluate(test_data_loader, model, loss_fn, device)

print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")

def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

emotion={
    0:"sadness",
    1:"joy",
    2:"love",
    3:"anger",
    4:"fear",
    5:"surprise"
}

text = "I hate doing Coursework!"

value,confidence=predict_sentiment(text, model, tokenizer, device)
print(f"Predicted emotion: {emotion[(value)]}, with confidence {confidence}")

text = "I actually like coursework"

value,confidence=predict_sentiment(text, model, tokenizer, device)
print(f"Predicted emotion: {emotion[(value)]}, with confidence {confidence}")

text = "I actually like coursework. I'm being sarcastic"

value,confidence=predict_sentiment(text, model, tokenizer, device)
print(f"Predicted emotion: {emotion[(value)]}, with confidence {confidence}")

text = "I'm shocked I haven't graduated yet"

value,confidence=predict_sentiment(text, model, tokenizer, device)
print(f"Predicted emotion: {emotion[(value)]}, with confidence {confidence}")

def get_model_and_evaluate(transformer_name):
    test_data = load_dataset("dair-ai/emotion", split="test", trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name,return_attention_mask=False, return_token_type_ids=False)
    test_data=test_data.map(tokenize)
    test_data = test_data.with_format(type="torch", columns=["ids", "label"])
    test_data_loader=DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    my_transformer = transformers.AutoModel.from_pretrained(transformer_name)
    my_model = Transformer(my_transformer, output_dim, freeze)
    my_model=my_model.to(device)
    my_model.load_state_dict(torch.load(transformer_name+".pt"))
    model_loss,model_accuracy = evaluate(test_data_loader, my_model, loss_fn, device)
    return model_loss,model_accuracy, my_transformer.config.hidden_size

bert_base_uncased_data=get_model_and_evaluate("bert-base-uncased")
bert_base_cased_data=get_model_and_evaluate("bert-base-cased")
bert_large_uncased_data=get_model_and_evaluate("bert-large-uncased")
bert_large_cased_data=get_model_and_evaluate("bert-large-cased")

model_names=["bert-base-uncased","bert-base-cased","bert-large-uncased","bert-large-cased"]
model_losses=[bert_base_uncased_data[0],bert_base_cased_data[0],bert_large_uncased_data[0],bert_large_cased_data[0]]
model_accs=[bert_base_uncased_data[1],bert_base_cased_data[1],bert_large_uncased_data[1],bert_large_cased_data[1]]
time=[338,355,812,865]
tranformer_hidden_dimension=[bert_base_uncased_data[2],bert_base_cased_data[2],bert_large_uncased_data[2],bert_large_cased_data[2]]

df_data={"Model Names":model_names,"Model Losses":model_losses,"Model Accuracies":model_accs,"Time taken to Fine-Tune the Model (secs)":time,"Tranformer Hidden Dimension":tranformer_hidden_dimension}
pd.DataFrame(df_data)
