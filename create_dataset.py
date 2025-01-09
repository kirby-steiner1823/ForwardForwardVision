from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json

transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root = './data',
                               train = True, download = True,
                               transform = transform)
mnist_loader = DataLoader(mnist_dataset, batch_size = 64, shuffle = True)

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_prompts(label):
    positive_prompt = f"Generate a positive description for digit {label}:"
    positive_input = tokenizer(positive_prompt,  return_tensors = "pt")

    positive_output = model.generate(**positive_input, max_length=20)
    positive_text = tokenizer.decode(positive_output[0], skip_special_tokens = True)

    negative_prompt = f"Generate a negative description for digit {label}:"
    negative_input = tokenizer(negative_prompt, return_tensors="pt")

    negative_output = model.generate(**negative_input, max_length=20)
    negative_text = tokenizer.decode(negative_output[0], skip_special_tokens=True)

    return positive_text, negative_text

dataset_with_text = []

for idx, (image, label) in enumerate(tqdm(mnist_dataset, desc="Generating Text Prompts")):
    positive_text, negative_text = generate_prompts(label)
    dataset_with_text.append((image, label, positive_text, negative_text))

with open("/data/train_data.json", "w") as f:
    json.dump(dataset_with_text, f)
