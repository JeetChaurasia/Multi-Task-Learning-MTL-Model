import pandas as pd
import numpy as np
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

# Define the maximum length of the input sequences
max_length = 200

# Define the labels for the aspect-based sentiment analysis task
aspect_labels = ['positive', 'negative', 'neutral']

# Define the labels for the emotion detection task
emotion_labels = ['Happy', 'sad', 'angry']

# Define the labels for the sentiment-emotion joint modeling task
sentiment_labels = ['positive', 'negative', 'neutral']

# Load the Semeval-2014 dataset
train_data = pd.read_csv('semeval2014_train.csv')
test_data = pd.read_csv('semeval2014_test.csv')

# Combine the training and testing data
data = pd.concat([train_data, test_data])

# Preprocess the text data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

data['review'] = data['review'].apply(preprocess_text)

# Load the spaCy model for dependency parsing
nlp = spacy.load('en_core_web_sm')

def extract_aspects(text):
    doc = nlp(text)
    aspects = []
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'FEATURE', 'ATTRIBUTE']:
            aspects.append(ent.text)
    return aspects

data['aspects'] = data['review'].apply(extract_aspects)

# Create a tokenizer to split the text into words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['review'])

X_train, X_test, y_train_aspect, y_test_aspect, y_train_emotion, y_test_emotion = train_test_split(data['review'], data['aspects'], data['emotion'], test_size=0.2, random_state=42)

# Create a TF-IDF vector izer to convert the text data into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define the multi-task learning model
input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=5000, output_dim=128, input_length=max_length)(input_layer)
lstm_layer = LSTM(64, dropout=0.2)(embedding_layer)

aspect_output = Dense(len(aspect_labels), activation='softmax')(lstm_layer)
emotion_output = Dense(len(emotion_labels), activation='softmax')(lstm_layer)

# Define the sentiment-emotion joint modeling component using PyTorch
class SentimentEmotionJointModel(nn.Module):
    def __init__(self):
        super(SentimentEmotionJointModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, len(sentiment_labels))
        self.fc3 = nn.Linear(64, len(emotion_labels))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        sentiment_output = self.fc2(x)
        emotion_output = self.fc3(x)
        return sentiment_output, emotion_output

joint_model = SentimentEmotionJointModel()

# Define the multi-task learning model with the sentiment-emotion joint modeling component
model = Model(inputs=input_layer, outputs=[aspect_output, emotion_output, joint_model(lstm_layer)])

model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mean_squared_error'], optimizer='adam', metrics=['accuracy'])

# Define the adversarial training component
class AdversarialTraining(nn.Module):
    def __init__(self):
        super(AdversarialTraining, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, len(sentiment_labels))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        sentiment_output = self.fc2(x)
        return sentiment_output

adversarial_model = AdversarialTraining()

# Define the adversarial loss function
def adversarial_loss(y_true, y_pred):
    return -torch.mean (y_true * torch.log(y_pred))

# Train the model with adversarial training
for epoch in range(5):
    for i, (x, y) in enumerate(DataLoader(X_train_tfidf, batch_size=32)):
        # Generate adversarial examples
        x_adv = x + 0.1 * torch.randn_like(x)
        y_adv = y

        # Train the model on the adversarial examples
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        outputs = model(x_adv)
        loss = adversarial_loss(y_adv, outputs)
        loss.backward()
        optimizer.step()

        # Train the model on the original examples
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        outputs = model(x)
        loss = model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mean_squared_error'], optimizer='adam', metrics=['accuracy'])
        loss.backward()
        optimizer.step()

# Evaluate the model
loss_aspect, accuracy_aspect, loss_emotion, accuracy_emotion, loss_sentiment, accuracy_sentiment = model.evaluate(X_test_tfidf, [y_test_aspect, y_test_emotion, y_test_sentiment])
print(f'Test accuracy (Aspect): {accuracy_aspect:.3f}')
print(f'Test accuracy (Emotion): {accuracy_emotion:.3f}')
print(f'Test accuracy (Sentiment): {accuracy_sentiment:.3f}')

# Save the model
model.save('multi_task_learning_model.h5')

# Load the saved model
loaded_model = torch.load('multi_task_learning_model.h5')

# Use the loaded model to make predictions
predictions = loaded_model.predict(X_test_tfidf)

# Evaluate the predictions
y_pred_aspect = np.argmax(predictions[0], axis=1)
y_pred_emotion = np.argmax(predictions[1], axis=1)
y_pred_sentiment = np.argmax(predictions[2], axis=1)

accuracy_aspect = accuracy_score(y_test_aspect, y_pred_aspect)
accuracy_emotion = accuracy_score(y_test_emotion, y_pred_emotion)
accuracy_sentiment = accuracy_score(y_test_sentiment, y_pred_sentiment)

print(f'Test accuracy (Aspect): {accuracy_aspect:.3f}')
print(f'Test accuracy (Emotion): {accuracy_emotion:.3f}')
print(f'Test accuracy (Sentiment): {accuracy_sentiment:.3f}')

f1_aspect = f1_score(y_test_aspect, y_pred_aspect, average='macro')
f1_emotion = f1_score(y_test_emotion, y_pred_emotion, average='macro')
f1_sentiment = f1_score (y_test_sentiment, y_pred_sentiment, average='macro')

print(f'Test F1 score (Aspect): {f1_aspect:.3f}')
print(f'Test F1 score (Emotion): {f1_emotion:.3f}')
print(f'Test F1 score (Sentiment): {f1_sentiment:.3f}')

auc_aspect = roc_auc_score(y_test_aspect, y_pred_aspect, multi_class='ovr')
auc_emotion = roc_auc_score(y_test_emotion, y_pred_emotion, multi_class='ovr')
auc_sentiment = roc_auc_score(y_test_sentiment, y_pred_sentiment, multi_class='ovr')

print(f'Test AUC score (Aspect): {auc_aspect:.3f}')
print(f'Test AUC score (Emotion): {auc_emotion:.3f}')
print(f'Test AUC score (Sentiment): {auc_sentiment:.3f}')

mse_aspect = mean_squared_error(y_test_aspect, y_pred_aspect)
mse_emotion = mean_squared_error(y_test_emotion, y_pred_emotion)
mse_sentiment = mean_squared_error(y_test_sentiment, y_pred_sentiment)

print(f'Test MSE score (Aspect): {mse_aspect:.3f}')
print(f'Test MSE score (Emotion): {mse_emotion:.3f}')
print(f'Test MSE score (Sentiment): {mse_sentiment:.3f}')

#ERROR handler

