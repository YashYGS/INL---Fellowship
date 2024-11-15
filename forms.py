from django import forms
from django.contrib.auth.forms import AuthenticationForm

class UserInfoForm(forms.Form):
    name = forms.CharField(label='Name', max_length=100)

class UploadFileForm(forms.Form):
    csv_file = forms.FileField(label='Upload CSV File')

class TaskForm(forms.Form):
    TASK_CHOICES = [
        ('plot', 'PLOT Existing Data'),
        ('mask', 'MASK Existing Data'),

    ]

    task = forms.ChoiceField(choices=TASK_CHOICES, widget=forms.RadioSelect(attrs={'class': 'task-button'}))

class CustomAuthenticationForm(AuthenticationForm):
    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get('username')
        password = cleaned_data.get('password')

        # Hardcoded credentials to allow access
        allowed_username = 'y'
        allowed_password = 'a'

        if str(username) != allowed_username or str(password != allowed_password):
            raise forms.ValidationError("Invalid username or password.")

        return cleaned_data