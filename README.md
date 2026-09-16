# MLOps2
# [Projet de mise en production d'un modèle de scoring]

## À propos

Ce projet a pour objectif de déployez et monitorez un modèle de scoring

## Table des matières

- 🪧 [À propos](#à-propos)
- 📦 [Prérequis](#prérequis)
- 🚀 [Installation](#installation)
- 🛠️ [Utilisation](#utilisation)
- 🤝 [Contribution](#contribution)
- 🏗️ [Construit avec](#construit-avec)
- 📝 [Authentification](#Authentification)
- 🛠️ [CI/CD](#CI/CD)
- 📚 [Documentation](#documentation)
- 🏷️ [Gestion des versions](#gestion-des-versions)
- 📝 [Licence](#licence)

## Prérequis

IL faut avoir d'installé sur la machine python et pip 

## Installation

Dans le dossier du projet via un terminal installer avec la commande pip install les lib suivante :
joblib==1.5.3
pandas==2.3.3
gradio==6.26.0
scikit-learn==1.9.0
imbalanced-learn==0.14.2
numpy==2.4.6
psycopg[binary]
evidently==0.7.21
pyarrow==25.0.1

ou sinon via pip install -r requirements.txt

## Utilisation

### déployer le projet sur un serveur local 
uv run python -m modele.main 

### lancement des tests en local
python -m pytest -q

### lancement tests avec rapport de couverture
python -m pytest --cov=modele --cov-branch --cov-report=term-missing --cov-report=html:htmlcov   

### obtenir les lib et leurs version actuellemnt installé dans le projet
python -m pip freeze     

## Contribution

Developpeur : N.Ulrick

## Construit avec

### Langages & Frameworks

Le langage python, le gestionnaire Git et gitflow, test écrit avec pytest

## Authentification

GitHub : avoir une clé ssh valide généré coté github
Render : avoir un coompte valide : login/mdp pour l'application et la bdd postgreSql sont fournie par render
Il suffit juste de lié GitHub à Render pour rendre le CD opérationel

## CI/CD

le fichier local de conf CI/CD se trouve dans le dossier workflows dans .github sous le nom ci-cd.yml
le fichier est autmatiquement détecté par github

/!\ attention le repository local doit être lié à votre repository distant sur github dans le cas
contraire il faut les relier via cette commande :
git remote add origin git@github.com:Username/repository_name.git ( via ssh )
ou
git remote add origin https://github.com/Username/repository_name.git ( via http )

## Documentation

Render : https://render.com
github : https://github.com/
Appli local : http://127.0.0.1:7860

## Gestion des versions

la gestion des version des versions et tag sont faite à la main via les commandes suivante :
git tag -a vX.X.X -m "Release vX.X.X"    
git push origin vX.X.X   

## Licence

Voir le fichier [LICENSE](./LICENSE.md) du dépôt.
