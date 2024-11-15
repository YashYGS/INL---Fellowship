from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# Create your views here.

from django.http import HttpResponse
from django.shortcuts import redirect
from .forms import UserInfoForm
from .forms import UploadFileForm
import os
import csv
from django.http import FileResponse, HttpResponseBadRequest
from django.core.files import File
import pandas as pd
import numpy as np

from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin

flag = False

class ProtectedView(LoginRequiredMixin, TemplateView):
    template_name = 'app2/welcome.html'


def protected_view(request):
    # View logic for the protected view
    return render(request, 'app2/welcome.html')



def choose_task(request):

    global flag

    # Check if the user is logged in (flag is True)
    if flag:

        default_csv_file_path = r'default.csv'  # Replace with the path to your default CSV file
        default_file_path = '/home/ysubrama/INLwebapp2/app2/test.csv'
        if request.method == 'POST':
            # Get the selected task (existing or new data)
            task = request.POST.get('task')

            if task == 'existing':
                csv_file_path = default_csv_file_path
                form = None
            else:
                form = UploadFileForm(request.POST, request.FILES)
                if form.is_valid():
                    csv_file = request.FILES['csv_file']
                    csv_file_path = handle_uploaded_file(csv_file)
                else:
                    csv_file_path = default_csv_file_path

            context = {
                'form': form,
                'task': task,
                'csv_file_path': csv_file_path,
                'default_file_path':default_file_path,
            }

            if 'submit' in request.POST:
                # Redirect to the respective task page (plot or train)
                if request.POST['submit'] == 'Plot':
                    return redirect('plot', csv_file_path=csv_file_path)
                elif request.POST['submit'] == 'Train':
                    return redirect('train', csv_file_path=csv_file_path)

            return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/choose_task.html', context)
        else:
            form = UploadFileForm()

        context = {
            'form': form,
            'task': None,
            'csv_file_path': default_csv_file_path,
            'default_file_path':default_file_path,
        }

        return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/choose_task.html', context)
    else:
        return redirect('welcome')

import base64
import matplotlib.pyplot as plt
import io


def plot(request, csv_file_path):
    global flag

    # Check if the user is logged in (flag is True)
    if flag:
        df = pd.read_csv(csv_file_path)
        columns = df.columns.tolist()

        if request.method == 'POST':
            x_column = request.POST['x_column']
            y_column = request.POST['y_column']

            # Create a new figure and axes object
            fig, ax = plt.subplots()

            # Plot the selected columns as scatter plot with improved appearance
            ax.scatter(df[x_column], df[y_column], s=5, edgecolors='k', linewidths=1, alpha=0.7)  # Decrease circle size
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(f'Plot of {y_column} against {x_column}')

            # Customize grid appearance (disable grid lines)
            ax.grid(False)

            # Customize ticks appearance


            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Convert the plot to a base64-encoded image string
            plot_image = base64.b64encode(buffer.read()).decode()
            buffer.close()

            # Save the plot to a file (optional, you can remove this part if not needed)
            output_file_path = r'outputPlot1.png'  # Replace with the desired output file path
            plt.savefig(output_file_path)

            # Close the figure to free up memory
            plt.close(fig)

            # Read the PNG file in binary mode and encode it as base64 (optional, you can remove this part if not needed)
            with open(output_file_path, 'rb') as file:
                encoded_image = base64.b64encode(file.read()).decode('utf-8')

            context = {
                'image': plot_image,
                'columns': columns,
            }

            return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/plot.html', context)

        context = {
            'columns': columns,
        }

        return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/plot.html', context)
    else:
        return redirect('welcome')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def about_us(request):
    return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/about_us.html')


from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout

@login_required
def protected_view(request):
    # View logic for the protected view
    return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/protected.html')

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

from django import forms
from django.contrib.auth.forms import AuthenticationForm


from .forms import CustomAuthenticationForm
from django.contrib.auth import authenticate, login
from django.contrib.sessions.models import Session


def welcome(request):

    global flag

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        string1 = "cynics"
        string2 = "yash@15"

        if username == string1 and password == string2:
            # Valid credentials, redirect to the next page (e.g., home page)
            flag = True
            return redirect('choose_task')
        else:
            # Invalid credentials, show error message
            error_message = "Invalid username or password."
            return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/welcome.html', {'error_message': error_message})

    return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/welcome.html')



def logout_view(request):
    logout(request)
    return redirect('signup_login')


def save_labels_to_csv(csv_file_path, labels):
    #output_file_path = os.path.splitext(csv_file_path)[0] + '_output.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label'])
        for label in labels:
            writer.writerow([label])


def train(request, csv_file_path):
    global flag

    # Check if the user is logged in (flag is True)
    if flag:
        if request.method == 'POST':
            algorithm = request.POST.get('algorithm')
            input_data = pd.read_csv(csv_file_path)
            output_labels = pd.read_csv(r'AnomLabels.csv')

             # Calculate the mean for each column


            if algorithm == 'algorithm1':

                # Merge the input and output data based on the index
                data = pd.concat([input_data, output_labels], axis=1)

                # Split the merged data into input features and output labels
                features = data.iloc[:, 1:]
                labels = data.iloc[:, -1]

                # Normalize the input features
                #scaler = StandardScaler()
                #normalized_features = scaler.fit_transform(features)

                # Convert the data to numpy arrays
                #normalized_features = np.array(normalized_features)
                #labels = np.array(labels)

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

                # Define the architecture of the neural network Actication function
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(500, activation='relu', input_shape=(features.shape[1],)),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])

                # Compile the model
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                # Define early stopping callback
                early_stopping = EarlyStopping(monitor='val_loss', patience=10)

                # Train the model with early stopping
                model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

                # Make predictions on the testing data
                predictions = model.predict(X_test)
                rounded_predictions = np.round(predictions)

                # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
                tn, fp, fn, tp = confusion_matrix(y_test, rounded_predictions).ravel()
                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)

                # Make predictions on the entire input data
                all_predictions = model.predict(features)
                all_rounded_predictions = np.round(all_predictions)

                label_file_path = '/home/ysubrama/INLwebapp2/app2/labelOutput.csv'
                # Save the labels to a CSV file
                save_labels_to_csv(label_file_path, all_rounded_predictions)

                context = {
                    'csv_file_path': csv_file_path,
                    'output_file' : label_file_path,
                    'false_positive_rate': fpr,
                    'true_positive_rate': tpr,
                    'running': False,
                }
                return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/train.html', context)

            elif algorithm == 'algorithm2':

                # Merge the input and output data based on the index
                data = pd.concat([input_data, output_labels], axis=1)

                # Split the merged data into input features and output labels
                features = data.iloc[:, :-1]
                labels = data.iloc[:, -1]

                # Normalize the input features
                #scaler = StandardScaler()
                #features = scaler.fit_transform(features)

                # Convert the data to numpy arrays
                features = np.array(features)
                labels = np.array(labels)

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

                # Create a decision tree classifier
                model = DecisionTreeClassifier()

                # Train the model
                model.fit(X_train, y_train)

                # Evaluate the model
                accuracy = model.score(X_test, y_test)

                # Calculate the predicted labels
                predicted_labels = model.predict(X_test)

                # Calculate the confusion matrix
                cm = confusion_matrix(y_test, predicted_labels)
                tn, fp, fn, tp = cm.ravel()

                # Calculate the false positive rate (FPR) and true positive rate (TPR)
                fpr = fp / (fp + tn)
                tpr = tp / (tp + fn)

                fpr = round(fpr, 2)
                tpr = round(tpr, 2)

                 # Make predictions on the entire input data
                all_predicted_labels = model.predict(features)
                label_file_path = '/home/ysubrama/INLwebapp2/app2/labelOutput.csv'

                # Save the labels to a CSV file
                save_labels_to_csv(csv_file_path, all_predicted_labels)

                context = {
                    'csv_file_path': csv_file_path,
                    'accuracy': accuracy,
                    'false_positive_rate': fpr,
                    'true_positive_rate': tpr,
                    'output_file' : label_file_path,

                }
                return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/train.html', context)

            elif algorithm == 'algorithm3':

                # Merge the input and output data based on the index
                data = pd.concat([input_data, output_labels], axis=1)

                # Split the merged data into input features and output labels
                features = data.iloc[:, :-1]
                labels = data.iloc[:, -1]



                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

                # Reshape the output labels to a 1-dimensional array
                y_train = np.ravel(y_train)

                # Create a Random Forest classifier with varying number of trees
                tree_counts = [100]  # Vary the number of trees as desired

                for n_estimators in tree_counts:
                    # Create a Random Forest classifier with the current number of trees
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=10)  # Adjust max_depth as needed

                    # Train the model
                    model.fit(X_train, y_train)

                    # Make predictions on the testing set
                    test_predictions = model.predict(X_test)

                    # Compute the accuracy of the model
                    accuracy = accuracy_score(y_test, test_predictions)

                    # Compute the confusion matrix
                    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()

                    # Compute the false positive rate (FPR) and true positive rate (TPR)
                    fpr = fp / (fp + tn)
                    tpr = tp / (tp + fn)

                    # Print the number of trees and corresponding accuracies, FPR, and TPR
                    fpr = round(fpr, 2)
                    tpr = round(tpr, 2)

                    print(f"Number of Trees: {n_estimators}, Accuracy: {accuracy:.4f}, FPR: {fpr:.4f}, TPR: {tpr:.4f}")

                 # Make predictions on the entire input data
                all_test_predictions = model.predict(features)

                label_file_path = '/home/ysubrama/INLwebapp2/app2/labelOutput.csv'

                 # Save the labels to a CSV file
                save_labels_to_csv(csv_file_path, all_test_predictions)

                context = {
                    'false_positive_rate': fpr,
                    'true_positive_rate': tpr,
                    'output_file' : label_file_path,

                }
                return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/train.html', context)

            else:
                return HttpResponse('Invalid algorithm selected')

        return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/train.html')
    else:
        return redirect('welcome')


def mask_options(request):
    global flag

    # Check if the user is logged in (flag is True)
    if flag:
        if request.method == 'POST':
            algorithm = request.POST.get('algorithm')

            # Perform necessary operations based on the selected algorithm
            if algorithm == 'algorithm1':
                default_dataset_path = r'default.csv'
                output_file_path = '/home/ysubrama/INLwebapp2/app2/maskOutput.csv'
                # Load the default dataset
                default_dataset = pd.read_csv(default_dataset_path)

                # Generate a random dataset of shape (7, 50)
                random_dataset = np.random.rand(7, 7)

               # Multiply the default dataset with the random dataset (excluding the first column)
                output = np.matmul(default_dataset.iloc[:, 1:].to_numpy(), random_dataset)

                # Create a DataFrame from the output numpy array
                output_df = pd.DataFrame(output)

                 # Add the 'time' column in front of the output data
                output_df.insert(0, 'time', range(len(output_df)))

                # Save the output as a CSV file
                output_df.to_csv(output_file_path, index=False)

                # Check if the output file was created successfully
                if os.path.exists(output_file_path):
                    # Pass the output file path to the template context
                    print(output_file_path)
                    print("#1")
                    context = {'output_file': output_file_path}
                    print(output_file_path)
                    print("#2")
                    return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/mask_options.html', context)
                else:
                    return HttpResponse('Failed to generate output file')

            elif algorithm == 'algorithm2':
                default_dataset_path = r'default.csv'
                output_file_path = '/home/ysubrama/INLwebapp2/app2/maskOutput.csv'

                # Load the default dataset
                default_dataset = pd.read_csv(default_dataset_path)

                # Generate a random dataset of shape (7, 7)
                random_dataset = np.random.rand(7, 50)

                # Multiply the default dataset with the random dataset (excluding the first column)
                output = np.matmul(default_dataset.iloc[:, 1:].to_numpy(), random_dataset)

                # Create a DataFrame from the output numpy array
                output_df = pd.DataFrame(output)

                 # Add the 'time' column in front of the output data
                output_df.insert(0, 'time', range(len(output_df)))

                # Save the output as a CSV file
                output_df.to_csv(output_file_path, index=False)

                # Check if the output file was created successfully
                if os.path.exists(output_file_path):
                    # Pass the output file path to the template context
                    context = {'output_file': output_file_path}
                    return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/mask_options.html', context)
                else:
                    return HttpResponse('Failed to generate output file')

            else:
                # Handle the case when an invalid algorithm is selected
                return HttpResponse('Invalid algorithm selected')

        # Render the mask_options template initially
        return render(request, '/home/ysubrama/INLwebapp2/app2/templates/app2/mask_options.html')
    else:
        return redirect('welcome')


def download_default_file(request):
    csv_file_path1 = r'default.csv'  # Replace with the path to your default CSV file

    output_file_path = csv_file_path1
    if not os.path.exists(output_file_path):
        return HttpResponseBadRequest("Error: Output file not found.")

    # Open the output PNG file
    with open(output_file_path, 'rb') as file:
        response = HttpResponse(File(file), content_type='file/csv')

    # Set the Content-Disposition header to force file download
    response['Content-Disposition'] = 'attachment; filename="output.csv"'

    return response


def download_file(request):
    #file_path = r'C:\Users\yasha\projects\django-web-app3\neuralNet\masking\input.csv'  # Replace with the path to your default CSV file
    csv_file_path2 = '/home/ysubrama/INLwebapp2/app2/maskOutput.csv'

    output_file_path = csv_file_path2
    if not os.path.exists(output_file_path):
        return HttpResponseBadRequest("Error: Output file not found.")

    # Open the output PNG file
    with open(output_file_path, 'rb') as file:
        response = HttpResponse(File(file), content_type='file/csv')

    # Set the Content-Disposition header to force file download
    response['Content-Disposition'] = 'attachment; filename="output.csv"'

    return response

def download_label(request):
    #file_path = r'C:\Users\yasha\projects\django-web-app3\neuralNet\masking\input.csv'  # Replace with the path to your default CSV file
    csv_file_path2 = '/home/ysubrama/INLwebapp2/app2/labelOutput.csv'

    output_file_path = csv_file_path2
    if not os.path.exists(output_file_path):
        return HttpResponseBadRequest("Error: Output file not found.")

    # Open the output PNG file
    with open(output_file_path, 'rb') as file:
        response = HttpResponse(File(file), content_type='file/csv')

    # Set the Content-Disposition header to force file download
    response['Content-Disposition'] = 'attachment; filename="label_output.csv"'

    return response



def handle_uploaded_file(file):
    file_path = file.name  # Replace with the desired file storage path
    with open(file_path, 'wb') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return file_path

