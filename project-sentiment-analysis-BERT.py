#!/usr/bin/env python3

#
#
#
#
#  Sample data source:  https://www.kaggle.com/datasets/vishweshsalodkar/customer-feedback-dataset/code
#  Original script:  https://www.kaggle.com/code/sharmageetika/text-cleaning-bert-sentimtal-analysis
#
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import string
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import nltk
from wordcloud import WordCloud
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# place all of your feedback / sentiment files into a central directory
# TODO - take as input, when using for review to focus on one person's data
#for dirname, _, filenames in os.walk('/home/tjrandall/Downloads/sentiment'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# TODO - take as input, when using for review to focus on one person's data
customer_feedback = pd.read_csv("/home/tjrandall/Downloads/sentiment/projFeedback.csv")

# define the data columns
customer_feedback[['Text', 'Sentiment', 'Source', 'Date/Time', 'User ID', 'Location', 'Confidence Score']] = customer_feedback['Text, Sentiment, Source, Date/Time, User ID, Location, Confidence Score'].str.split(',', expand = True)
# remove the headers
customer_feedback = customer_feedback.drop('Text, Sentiment, Source, Date/Time, User ID, Location, Confidence Score', axis=1)
customer_feedback.head()

def clean_text(text: pd.Series) -> pd.Series:
    """
    Function cleans the text column by:
    - Converting text to lowercase
    - Removing special characters, digits
    - Removing extra spaces

    Parameters
    ----------
    text : pd.Series
        Text column (pandas Series of strings, e.g., reviews).

    Returns
    -------
    text : pd.Series
        Column with cleaned text.
    """
    # Ensure text is in lowercase
    text = text.str.lower()

    # Remove any special characters and digits
    text = text.str.replace(r'[^a-z\s]', '', regex=True)

    # Remove extra spaces between words
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()

    return text

# Generate word clouds
def plot_wordcloud(text, title, stopwords):
    """
    Function creates a wordcloud

    Parameters
    ----------
    text : pd.Series
        Text column (pandas Series of strings, e.g., reviews).

    Returns
    -------
    N/a
    """
    if not text.strip():
        print(f"No words to generate a word cloud for: {title}")
        return

    wordcloud = WordCloud(
        width=800, height=400,
        background_color='black',
        stopwords=stopwords,
        colormap='viridis'
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

def create_synonym_dict():
    """
    Function creates positive and negative synonyms
    dictionary

    Parameters
    ----------
    None

    Returns
    -------
    Array : synonym_dict

    """
    pos_list=set(opinion_lexicon.positive())
    neg_list=set(opinion_lexicon.negative())

    synonym_dict = {
        'Positive':pos_list,
        'Negative':neg_list
    }

    return synonym_dict

def predict_user_input(user_input):
    """
    Function to make predictions on a single user input text

    Parameters
    ----------
    Text: user_input

    Returns
    -------
    Array : predicted_class

    """
    # Tokenize the user input text
    inputs = tokenizer(
        user_input,                       # Single text input
        padding=True,                      # Pad to max length
        truncation=True,                   # Truncate if length exceeds max_length
        return_tensors="pt",               # Return PyTorch tensors
        max_length=512                     # Ensure max length for BERT
    )

    # Move inputs to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Set the model to evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()  # Get the predicted class index

    # Map the predicted label to the corresponding sentiment
    predicted_class = 'Positive' if predicted_label == 1 else 'Negative'

    return predicted_class

# Assuming 'customer_feedback' is your DataFrame and 'Text' is the column to clean
customer_feedback['Cleaned_Text'] = clean_text(customer_feedback['Text'])
customer_feedback['Sentiment'] = customer_feedback['Sentiment'].str.strip()
customer_feedback = customer_feedback.dropna()
customer_feedback

sentiment_counts = customer_feedback['Sentiment'].value_counts()

# Plotting the sentiment counts as a bar chart
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'green'])

# Adding titles and labels
plt.title('Sentiment Counts')
plt.xlabel('Sentiment')
plt.ylabel('Counts')

# Displaying the plot
plt.show()

# Separate positive and negative reviews
positive_text = ' '.join(customer_feedback[customer_feedback['Sentiment'] == 'Positive']['Cleaned_Text'])
negative_text = ' '.join(customer_feedback[customer_feedback['Sentiment'] == 'Negative']['Cleaned_Text'])

# Stopwords
stop_words = set(stopwords.words('english'))

# Plot word clouds for positive and negative reviews
plot_wordcloud(positive_text, "Positive Word Cloud", stop_words)
plot_wordcloud(negative_text, "Negative Word Cloud", stop_words)

# Analysing Number of Positive and Negative Feedback Per Source
# Grouping data by 'Source' and 'Sentiment' and counting occurrences
sentiment_counts = customer_feedback.groupby(['Source', 'Sentiment']).size().unstack(fill_value=0)

# Plotting the sentiment counts as a bar chart
sentiment_counts.plot(kind='bar', stacked=False, color=['green', 'red'], figsize=(8, 6))

# Adding titles and labels
plt.title('Sentiment Counts per Source')
plt.xlabel('Source')
plt.ylabel('Counts')
plt.legend(title='Sentiment', labels=['Positive', 'Negative'])

# Displaying the plot
plt.tight_layout()
plt.show()

# Analysing Number of Positive and Negative Feedback Per Location
# Grouping data by 'Location' and 'Sentiment' and counting occurrences
sentiment_counts = customer_feedback.groupby(['Location', 'Sentiment']).size().unstack(fill_value=0)

# Plotting the sentiment counts as a bar chart
sentiment_counts.plot(kind='bar', stacked=False, color=['green', 'red'], figsize=(8, 6))

# Adding titles and labels
plt.title('Sentiment Location per Source')# Ensure the column is in datetime format
customer_feedback['Time'] = pd.to_datetime(customer_feedback['Date/Time'])

# Extract the start of the hour (only the hour component)
customer_feedback['Hour'] = customer_feedback['Time'].dt.hour

# Analysing Number of Positive and Negative Feedback Per Hour
# Grouping data by 'Hour' and 'Sentiment' and counting occurrences
sentiment_counts = customer_feedback.groupby(['Hour', 'Sentiment']).size().unstack(fill_value=0)

# Plotting the sentiment counts as a bar chart
sentiment_counts.plot(kind='bar', stacked=False, color=['green', 'red'], figsize=(8, 6))

# Adding titles and labels
plt.title('Sentiment Location per Source')
plt.xlabel('Location')
plt.ylabel('Counts')
plt.legend(title='Sentiment', labels=['Positive', 'Negative'])

# Displaying the plot
plt.tight_layout()
plt.show()
plt.xlabel('Location')
plt.ylabel('Counts')
plt.legend(title='Sentiment', labels=['Positive', 'Negative'])

# Displaying the plot
plt.tight_layout()
plt.show()


#---------------------    BERT -----------
# Define a dictionary of synonyms for positive and negative words
#
# original synonym dictionary - use this for initial runs for
# faster script processing.
synonym_dict = {
    'Positive': ['it is good', 'great', 'fantastic', 'awesome', 'love', 'amazing', 'wonderful', 'excellent'],
    'Negative': ['bad', 'horrible', 'terrible', 'awful', 'hate', 'disappointing', 'poor', 'lousy']
}

#  Use the function call for a more robust  Positive and Negative dictionary
# synonym_dict = create_synonym_dict()

customer_feedback = customer_feedback[['Sentiment', 'Cleaned_Text']]

# Create a list to store the data
df = []

# Add positive examples to the data list
for text in synonym_dict['Positive']:
    df.append([text, 'Positive'])

# Add negative examples to the data list
for text in synonym_dict['Negative']:
    df.append([text, 'Negative'])

# Create a DataFrame from the data list
df = pd.DataFrame(df, columns=['Cleaned_Text', 'Sentiment'])

# Append the newly created DataFrame to the original customer_feedback DataFrame
customer_feedback = pd.concat([customer_feedback, df], ignore_index=True)

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    customer_feedback['Cleaned_Text'], customer_feedback['Sentiment'], test_size=0.2, random_state=42
)

# Map the labels to numerical values (0 for Negative, 1 for Positive)
train_labels = train_labels.map({'Negative': 0, 'Positive': 1})
test_labels = test_labels.map({'Negative': 0, 'Positive': 1})

# Tokenize the training data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_inputs = tokenizer(
    list(train_texts),  # Ensure train_texts is a list
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=512
)

# Tokenize testing data
test_inputs = tokenizer(
    list(test_texts),  # Ensure test_texts is a list
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=512
)

# Define the number of labels for classification
num_labels = 2

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Convert tokenized inputs and labels into a TensorDataset for training
train_dataset = TensorDataset(
    train_inputs['input_ids'],
    train_inputs['attention_mask'],
    torch.tensor(train_labels.values, dtype=torch.long)  # Ensure labels are torch tensors of integers
)

# Create DataLoader for training
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Create DataLoader for testing
test_dataset = TensorDataset(
    test_inputs['input_ids'],
    test_inputs['attention_mask'],
    torch.tensor(test_labels.values, dtype=torch.long)  # Ensure test labels are torch tensors
)

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)

# Initialize variables for early stopping and training loss tracking
best_train_loss = float('inf')
counter = 0
patience = 3  # Number of epochs without improvement before early stopping
epochs = 6

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_train_loss = 0

    for batch in train_dataloader:  # Iterate through the DataLoader
        input_ids, attention_mask, labels = batch  # Unpack the batch into individual components

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)

    # Print the average training loss for this epoch
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}")

    # Early stopping based on training loss
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping due to no improvement in training loss.")
            break

# Set the device to CUDA if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device (GPU or CPU)
model.to(device)

# Set the model to evaluation mode
model.eval()

# Initialize lists for storing true labels and predicted labels
all_labels = []
all_preds = []

correct_predictions = 0
total_predictions = 0

# Disable gradient calculation for inference
with torch.no_grad():
    for batch in test_dataloader:
        # Unpack the batch correctly (tuple of input_ids, attention_mask, labels)
        input_ids, attention_mask, labels = batch

        # Move tensors to the same device as the model
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

        # Collect true labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted_labels.cpu().numpy())

        # Calculate accuracy
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix - a table used to visually evaluate the performance
# of a classification model by comparing its predicted class labels against
# the actual class labels
#
# Structure:
# It is a grid where the rows represent the actual classes and the columns
# represent the predicted classes.
#
# Components:
#  True Positive (TP): Correctly predicted positive cases.
#  True Negative (TN): Correctly predicted negative cases.
#  False Positive (FP): Incorrectly predicted positive cases (a "false alarm").
#  False Negative (FN): Incorrectly predicted negative cases (a "missed case").

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate and print accuracy
accuracy = correct_predictions / total_predictions
print(f"Test Accuracy: {accuracy:.4f}")

# Generate the classification report
print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))

# Example user input for prediction
user_input = input("Enter text for sentiment classification: ")

# Predict the sentiment for the user input
predicted_class = predict_user_input(user_input)

# Display the result
print(f"Predicted Sentiment: {predicted_class}")
