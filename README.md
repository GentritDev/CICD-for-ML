# Customer Support Ticket Classifier

 **Live Demo:** [huggingface.co/spaces/Genti123/Customer-Support-Classifier](https://huggingface.co/spaces/Genti123/Customer-Support-Classifier)
 
A machine learning system that automatically classifies customer support tickets into categories. Built with a full CI/CD pipeline using GitHub Actions and deployed on Hugging Face Spaces.

## Architecture

<img width="1416" height="1001" alt="image" src="https://github.com/user-attachments/assets/e3d614bf-808b-48e7-a783-4bcc0605b945" />


Every push to `main` triggers the CI pipeline which trains and evaluates the model. Once CI completes successfully, the CD pipeline automatically uploads the updated model and app to Hugging Face Spaces.

## How It Works

1. A customer support ticket is entered as free text
2. The text is vectorized using TF-IDF
3. A trained Scikit-learn classifier predicts the ticket category
4. The result is displayed instantly in the Gradio UI

## Tech Stack

| Layer | Technology |
|---|---|
| Model | Scikit-learn (TF-IDF + Classifier) |
| Serialization | skops |
| App | Gradio 5 |
| CI/CD | GitHub Actions |
| Hosting | Hugging Face Spaces |

## CI/CD Pipeline

**Continuous Integration** — triggered on every push to `main`:
- Installs dependencies
- Formats code with Black
- Trains the model on the dataset
- Evaluates and posts metrics as a PR comment

**Continuous Deployment** — triggered automatically after CI succeeds:
- Uploads the trained model to Hugging Face
- Deploys the updated Gradio app

## Ticket Categories

- Technical Issue
- Refund Request  
- Account Access

## Repository

[github.com/GentritDev/CICD-for-ML](https://github.com/GentritDev/CICD-for-ML)
