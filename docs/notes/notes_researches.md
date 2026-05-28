


# Anomaly Detection

- **Temporal Fusion Transformer (TFT)** : Excellente architecture basée sur l'attention, capable d'intégrer des variables externes (comme la vitesse de broche ou le type de matériau usiné) en plus du signal du capteur. : https://arxiv.org/abs/1912.09363

- **LSTM-AE (LSTM Autoencoder)** : Un classique très robuste pour les données de capteurs (vibrations, acoustique). Parfait pour repérer des anomalies dans des séquences temporelles sans avoir besoin de labels. : https://www.striim.com/blog/lstm-autoencoder-anomaly-detection/

- **Transformers-AE (Transformer Autoencoder)** : L'évolution du LSTM-AE. Utilise le mécanisme d'attention (au lieu de la récurrence) pour compresser et reconstruire le signal. Mieux adapté que le LSTM pour capter des dépendances très lointaines dans le signal (ex: un choc à t=1s qui a des répercussions à t=150s) et souvent plus rapide à entraîner.

- **Autoencodeur 1D-CNN** : Très efficace pour analyser les fréquences directement depuis le signal temporel brut (ou une FFT). Les convolutions 1D agissent comme des filtres fréquentiels apprenables. Souvent plus léger et plus rapide que les LSTM/Transformers pour la détection pure de signatures vibratoires.

- **TSFEL (Fenêtre glissante) + Isolation Forest** : L'approche "Machine Learning classique" (non Deep Learning). TSFEL extrait statistiquement les caractéristiques (features) sur des blocs de temps (fenêtres), et l'Isolation Forest isole les blocs anormaux. Très explicable, rapide à mettre en place, parfait comme "baseline" (modèle de référence) pour prouver que le concept fonctionne avant de sortir l'artillerie lourde.

- **CNN-LSTM Autoencoder** :



# Data forecasting
***Concept***: *Predict the next phase torque speed , because we know it , if it defers from the real next one , it could show that the "lames" are damaged or worn



## 1. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting (TFT)

* **Architecture :** Modèle hybride complexe combinant des réseaux récurrents (**LSTM**) pour capturer les dynamiques temporelles à court terme (locales) et des mécanismes d'auto-attention (**Transformers**) pour modéliser les dépendances et motifs à long terme.
* **Points clés :**
    * **Filtre intelligent (VSN - Variable Selection Networks) :** Évalue et sélectionne dynamiquement les variables les plus pertinentes à chaque instant $t$. Il permet d'"éteindre" l'influence des capteurs qui n'apportent que du bruit ou des parasites électriques.
    * **Interprétabilité native :** Permet d'extraire des scores d'importance pour savoir exactement quel capteur (ex: vibration axe Z, émission acoustique) et quel instant passée ont le plus influencé la prédiction.
    * **Prédictions probabilistes (Multi-quantiles) :** Le modèle ne prédit pas une valeur unique, mais des intervalles de confiance (ex: quantiles P10, P50, P90).
* **Application au diagnostic (Analyse des résidus) :** Le TFT génère une **bande de tolérance dynamique** (élastique) qui s'adapte au contexte d'usinage. Tant que le signal réel (le couple mesuré) reste à l'intérieur de cet intervalle de confiance, l'outil est considéré comme sain. Dès que la courbe réelle sort de la bande, une anomalie ou une usure critique est détectée.

---

## 2. NBEATSX (Neural Basis Expansion Analysis with Exogenous variables)

* **Architecture :** Modèle basé exclusivement sur un empilement de réseaux de neurones denses (**blocs MLP**) connectés par des liaisons résiduelles. Il évite la lourdeur de calcul des Transformers et des RNN, ce qui le rend beaucoup plus léger et rapide à entraîner.
* **Points clés :**
    * **Décomposition structurelle du signal :** Force mathématiquement ses blocs internes à séparer le signal en deux composantes hautement compréhensibles par l'humain :
        * **La Tendance (Trend) :** L'évolution de fond, la dérive lente du signal.
        * **La Saisonnalité (Seasonality) :** Les variations cycliques et les motifs répétitifs (ex: liés à la rotation de la broche).
    * **Variables exogènes (le "x") :** Intègre directement les paramètres de coupe connus à l'avance et issus du G-code (vitesse d'avance programmée, profondeur de passe, vitesse de rotation) pour ajuster instantanément sa prédiction de couple.
* **Application au diagnostic (Analyse des résidus) :** Idéal pour isoler précisément le type de défaillance. Une augmentation progressive de l'erreur sur la **Tendance** traduit une usure graduelle (augmentation du frottement). À l'inverse, une explosion soudaine de l'erreur sur la **Saisonnalité** signale une anomalie vibratoire ou une rupture partielle (écaillage d'une dent). 
    *Note : Contrairement au TFT, il effectue une prédiction ponctuelle exacte ; il est donc nécessaire de calculer soi-même un seuil d'alarme statistique (ex: $+3\sigma$ sur l'erreur de l'outil neuf).*

**3.PATCHTST** : 



# Data pre-processing


| Étape | Ce qu'il faut faire (Actions & Méthodes) | Pourquoi (Objectifs & Bénéfices) |
| :--- | :--- | :--- |
| **1. Nettoyage des données (Data Cleaning)** | **Gérer les valeurs manquantes :** Utiliser des techniques d'imputation (moyenne, médiane ou k-plus proches voisins).<br><br>**Gérer les valeurs aberrantes (outliers) :** Les détecter via des méthodes statistiques (Z-score, écart interquartile/IQR) puis les supprimer ou les remplacer par interpolation. | Les données brutes des capteurs contiennent des bruits, des erreurs de transmission ou de communication. Ces anomalies dégradent fortement les performances des modèles de Deep Learning. |
| **1.1 Lissage (Smoothing)** | Filtrer le bruit haute fréquence et éliminer les points aberrants (outliers). | Moyenne mobile, filtre passe-bas (ex: Butterworth) ou Savitzky-Golay |
| **2. Normalisation / Mise à l'échelle (Scaling)** | Amener toutes les données à une échelle commune en utilisant des techniques comme la **mise à l'échelle Min-Max** (ex: données comprises entre 0 et 1) ou la **standardisation** (moyenne à zéro et variance unitaire). | Permet de comparer et combiner des données issues de sources différentes. Cela **améliore la convergence et la stabilité** des modèles de Deep Learning lors de leur entraînement. |
| **3. Extraction de caractéristiques (Feature Extraction)** | Extraire des caractéristiques métier spécifiques à partir des signaux bruts :<br>• **Temporelles :** Moyenne, écart-type, valeur RMS, pics.<br>• **Fréquentielles :** Transformée de Fourier, ondelettes.<br>• **Temps-fréquence :** Transformée de Fourier à court terme (STFT). | Bien que le Deep Learning puisse apprendre seul, cette étape capture la dynamique globale et le contenu spectral des signaux, ce qui **améliore les performances et l'interprétabilité** du modèle. |
| **4. Sélection des caractéristiques (Feature Selection)** | Filtrer et classer les caractéristiques extraites précédemment en utilisant des méthodes comme l'**analyse de corrélation**, l'**information mutuelle** ou les **méthodes *wrapper***. | Permet d'identifier uniquement les données les plus pertinentes pour la tâche. Cela **réduit la dimensionnalité**, **limite le surapprentissage** (overfitting) et rend le modèle plus facile à comprendre. |
| **5. Séparation des données (Data Splitting)** | Diviser le jeu de données final, propre et transformé, en trois sous-ensembles distincts : **Entraînement** (Train), **Validation**, et **Test**. | • **Entraînement :** Pour que le modèle apprenne.<br>• **Validation :** Pour ajuster les réglages (hyperparamètres) et suivre les performances en cours de route.<br>• **Test :** Pour évaluer la performance finale du modèle sur des données qu'il n'a jamais vues. |





*Source* : 
- *https://www.researchgate.net/publication/393333605_Comparison_of_deep_learning_models_for_predictive_maintenance_in_industrial_manufacturing_systems_using_sensor_data*
- *https://link.springer.com/article/10.1007/s00170-025-15472-4*

- *Gemini : Texte reformulation*


