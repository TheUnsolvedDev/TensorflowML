import tensorflow as tf
import numpy as np
from config import MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM  # Ensure these are correct
from dataset import custom_standardization

# Define custom DenseEmbedding to load the model correctly


class DenseEmbedding(tf.keras.layers.Embedding):
    """Embedding layer that forces dense outputs to avoid IndexedSlices in gradients."""

    def call(self, inputs):
        return tf.convert_to_tensor(super().call(inputs))


def load_vectorizer(vocab_path='vocab.txt', sequence_length=MAX_LEN):
    with open(vocab_path, "r") as f:
        vocab = [line.strip() for line in f.readlines()]

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=len(vocab),
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize=custom_standardization
    )
    vectorizer.set_vocabulary(vocab)
    return vectorizer


def predict(texts, model_path='model.keras'):
    # Load model with custom layer
    model = tf.keras.models.load_model(model_path, custom_objects={
                                       "DenseEmbedding": DenseEmbedding})

    # Load vectorizer
    vectorizer = load_vectorizer()

    # Preprocess and vectorize
    if isinstance(texts, str):
        texts = [texts]
    text_tensor = tf.constant(texts)
    vecs = vectorizer(text_tensor)

    # Predict
    preds = model.predict(vecs)
    for text, pred in zip(texts, preds):
        label = "Fake" if pred[0] >= 0.5 else "Real"
        print(f"\nInput: {text}\nPrediction: {label} ({pred[0]:.4f})")


# Example usage
if __name__ == '__main__':
    test_samples = [
        "Congratulations, you've won a $1,000 gift card! Please click the link to claim your prize.",
        "Dear user, your bank account has been compromised. Please provide your login details to verify your identity immediately.",
        "I wanted to follow up on our last meeting and discuss the next steps for the project. Let me know when you're available.",
        "Attached is the report for the quarterly review. Please go through it and let me know if you have any questions.",
        "You are being selected for a limited time investment opportunity! This could earn you millions. Sign up now!",
        "I wanted to follow up on our last meeting and discuss the next steps for the project. Let me know when you're available.",
        "This is a reminder for the upcoming team meeting tomorrow at 10 AM. Looking forward to your insights on the new project.",
        "Just checking in to see how things are going with the new system setup. Any feedback would be helpful.",
        "I'm reaching out to confirm our lunch meeting at 1 PM today. Looking forward to discussing the new proposal.",
        "Amazon.com Password Assistance Greetings from Amazon.com. To finish resetting your password jarnold@enron.com, please visit our site using one of the personalized links below. The following link can be used to visit the site using the secure server: <URL> The following link can be used to visit the site using the standard server: <URL> It's easy. Simply click on one of the links above to return to our Web site. If this doesn't work, you may copy and paste the link into your browser's address window, or retype it there. Once you have returned to our Web site, you will be given instructions for resetting your password. If you have any difficulty resetting your password, please feel free to contact us by responding to this e-mail. Thank you for visiting Amazon.com! Amazon.com Earth's Biggest Selection <URL>  1"
    ]

    predict(test_samples)
