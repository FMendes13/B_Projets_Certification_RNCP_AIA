pipeline {
    agent any

    environment {
        VENV = 'venv'
    }

    stages {
        stage('Clone Repo') {
            steps {
                git url: 'https://github.com/<TON_UTILISATEUR>/<TON_REPO>.git', branch: 'main'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'python3 -m venv $VENV'
                sh '. $VENV/bin/activate && pip install --upgrade pip'
                sh '. $VENV/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                sh '. $VENV/bin/activate && pytest tests/'
            }
        }

        stage('Evaluate Model') {
            steps {
                sh '. $VENV/bin/activate && python model/evaluate_model.py'
            }
        }

        stage('Success') {
            steps {
                echo '✅ Pipeline terminée avec succès !'
            }
        }
    }

    post {
        failure {
            echo '❌ Pipeline échouée.'
        }
    }
}

