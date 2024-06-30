import os
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import pickle
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Inception-V3 model
model = inception_v3(pretrained=True)
model.eval()
model = model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().cpu().numpy()

def create_feature_matrix(dataset_folder):
    feature_matrix = []
    image_paths = []
    for image_name in os.listdir(dataset_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dataset_folder, image_name)
            features = extract_features(image_path)
            feature_matrix.append(features)
            image_paths.append(image_path)
    return np.array(feature_matrix), image_paths

def save_feature_matrix(feature_matrix, image_paths, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump({'feature_matrix': feature_matrix, 'image_paths': image_paths}, f)

def load_feature_matrix(input_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data['feature_matrix'], data['image_paths']

def find_similar_images(query_features, feature_matrix, image_paths, top_k=5):
    similarities = cosine_similarity([query_features], feature_matrix)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(image_paths[i], similarities[i]) for i in top_indices]

def plot_similar_images(query_image_path, similar_images):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    query_image = Image.open(query_image_path)
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    for i, (image_path, similarity) in enumerate(similar_images, start=1):
        image = Image.open(image_path)
        axes[i].imshow(image)
        axes[i].set_title(f"Similarity: {similarity:.4f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Image Similarity Search")
    parser.add_argument("--mode", choices=['create', 'search'], required=True, help="Mode: create feature matrix or search similar images")
    parser.add_argument("--dataset", help="Path to dataset folder")
    parser.add_argument("--output", help="Path to save/load feature matrix")
    parser.add_argument("--query", help="Path to query image")
    args = parser.parse_args()

    if args.mode == 'create':
        if not args.dataset or not args.output:
            print("Please provide dataset folder and output file path.")
            return

        feature_matrix, image_paths = create_feature_matrix(args.dataset)
        save_feature_matrix(feature_matrix, image_paths, args.output)
        print(f"Feature matrix saved to {args.output}")

    elif args.mode == 'search':
        if not args.output or not args.query:
            print("Please provide feature matrix file and query image path.")
            return

        feature_matrix, image_paths = load_feature_matrix(args.output)
        query_features = extract_features(args.query)
        similar_images = find_similar_images(query_features, feature_matrix, image_paths)

        print("Similar images:")
        for path, similarity in similar_images:
            print(f"Image: {path}, Similarity: {similarity:.4f}")

        plot_similar_images(args.query, similar_images)

if __name__ == "__main__":
    main()
