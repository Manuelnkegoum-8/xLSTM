import torch
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_epoch(model, dataloader, optimizer, criterion, device):
    avg_loss, avg_acc = 0.0, 0.0
    n = 0.0
    model.train()
    
    # Wrap dataloader with tqdm for a progress bar
    with tqdm(total=len(dataloader), desc="Training", colour='blue',ncols=100) as pbar:
        for data in dataloader:
            texts, labels = data
            bs = texts.size(0)
            texts = texts.to(device)
            labels = labels.to(device)
            preds = model(texts)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item() * bs
            n += bs
            
            # Update the progress bar with the current loss
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
        
    avg_loss /= n
    #logging.info(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def validate(model, dataloader, criterion, device,tokenizer):
    avg_loss, avg_acc = 0.0, 0.0
    n = 0.0
    model.eval()
    # Wrap dataloader with tqdm for a progress bar
    with tqdm(total=len(dataloader), desc="Validation", colour='yellow',ncols=100) as pbar:
        for i, data in enumerate(dataloader):
            texts, labels = data
            bs = texts.size(0)
            texts = texts.to(device)
            labels = labels.to(device)
            
            preds = model(texts)
            loss = criterion(preds, labels)
            
            avg_loss += loss.item() * bs
            n += bs
            if i==0:
                debug(preds,texts,labels,tokenizer,num_samples=min(bs,4))
            # Update the progress bar with the current loss
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
        
    avg_loss /= n
    #logging.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def debug(preds,texts,labels,tokenizer,num_samples = 4):
    preds = preds.argmax(dim=-1)  # Assuming classification task
    for j in range(num_samples):
        input_text = tokenizer.decode(texts[j], skip_special_tokens=True)
        predicted_text = tokenizer.decode(preds[j], skip_special_tokens=True)
        true_text = tokenizer.decode(labels[j], skip_special_tokens=True)
                    
        logging.info(f"\nSample {j}:")
        logging.info(f"Input: {input_text}")
        logging.info(f"Prediction: {predicted_text}")
        logging.info(f"True Label: {true_text}")