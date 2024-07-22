
# MakeMoreSmiles

Welcome to the MakeMoreSmiles project! This project utilizes Streamlit to create a web application for generating and analyzing SMILES (Simplified Molecular Input Line Entry System) strings using a pretrained and finetuned model using PyTorch.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
   - [Local Setup](#local-setup)
   - [Docker Setup](#docker-setup)
4. [Usage](#usage)


## Overview

MakeMoreSmiles is a tool designed to generate and analyze chemical structures represented as SMILES strings. The application leverages PyTorch for model inference and RDKit for chemical informatics and molecule drawing. The web interface is built using Streamlit.

## Features

- Generate SMILES strings using a fine-tuned model.
- Validate and analyze generated SMILES strings.
- Visualize molecular structures.

## Installation

### Local Setup

To set up the project locally, follow these steps:

1. **Clone the Repository:**

   ```sh
   git clone https://github.com/Nickspizza001/makemoresmiles.git
   cd makemoresmiles

   ```


2. **Install Dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Application:**

   ```sh
   streamlit run app.py
   ```

### Docker Setup

To run the application using Docker, follow these steps:

1. **Build the Docker Image:**

   ```sh
   docker build -t makemoresmiles .
   ```

2. **Run the Docker Container:**

   ```sh
   docker run -p 8501:8501 makemoresmiles
   ```

3. **Access the Application:**

   Open your web browser and navigate to `http://0.0.0.0:8501`.

### Streamlit Cloud Setup

The application is also hosted on Streamlit Cloud. You can access it at:

[MakeMoreSmiles on Streamlit](https://makemoresmiles.streamlit.app/)

## Usage

1. **Generate SMILES:**

   - Open the application in your web browser.
   - Enter the required input parameters for SMILES generation.
   - Click the "Generate" button to produce new SMILES strings.

2. **Analyze SMILES:**

   - The generated SMILES strings will be displayed.
   - You can visualize the molecular structures and validate them.



---
