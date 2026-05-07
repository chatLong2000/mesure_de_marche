import numpy as np
import cv2
from scipy.signal import find_peaks

class SignalClassifier():
    def __init__(self, height_threshold=10):
        self.height_threshold = height_threshold

    def crop_to_square(self, img):
        """ 
        Recadre une image pour qu'elle soit carrée.

        Args:
            img : Image à recadrer (2D).
        
        Returns:
            img : Image recadrée (2D).
        """
        h, w = img.shape

        # Si l'image est déjà carrée
        if h == w:
            return img 
        
        # Cas où la hauteur est supérieure à la largeur
        if h > w:
            start = (h - w) // 2
            return img[start:start+w, :]
        else:
            start = (w - h) // 2
            return img[:, start:start+h]

    def subtract_median(self, images):
        """
        Soustrait la médiane des 3 premières images de la 4ème image.

        Args:
            images : Liste de 4 images (3D).

        Returns:
            result : Image résultante (2D).
        """
        assert len(images) == 4, "Il faut exactement 4 images"
        
        # Mediane des 3 premières images
        median_image = np.median(np.stack(images[:3]), axis=0)

        # Soustraction de la médiane de la 4ème image
        result = images[3] - median_image
        
        # Clippage (0-255) et conversion en uint8
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Recadrage pour obtenir une image carrée
        result = self.crop_to_square(result)
        
        # Seuillage pour éliminer les pixels faibles
        result[result < 30] = 0
        return result
    
    def extract_perpendicular_vector(self, img, alpha=0.99, nb_pixels=15):
        """
        Extrait un vecteur d'intensité de pixels le long d'une ligne perpendiculaire
        au vecteur (centre -> tip), centrée à `alpha`% de ce vecteur, sur `img`.
        
        Args:
            img : Image en niveaux de gris (2D).
            alpha : Proportion entre centre et tip (0.0 = centre, 1.0 = tip).
            nb_pixels : Nombre de pixels avant/après l'intersection (total = 2*nb_pixels + 1).
        
        Returns:
            line_points : Liste des intensités de pixels le long de la ligne.
            valid : Booléen indiquant si la ligne est valide (True) ou non (False).
        """

        # Seuillage pour obtenir une image binaire
        _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        # Trouver les pixels non nuls
        ys, xs = np.where(img_thresh > 0)
        coords = np.column_stack((xs, ys))
        
        h, w = img.shape
        # Centrée de l'image
        center = np.array([w // 2, h // 2])
        # Calculer la distance entre chaque pixel et le centre
        dists = np.linalg.norm(coords - center, axis=1)
        # Trouver le pixel le plus éloigné du centre
        tip = coords[np.argmax(dists)]

        vec = tip - center
        new_base = center + alpha * vec

        # Calculer le vecteur unitaire perpendiculaire au vecteur (centre -> tip)
        d_perp = np.array([-vec[1], vec[0]], dtype=float)
        d_perp = d_perp / np.linalg.norm(d_perp)

        line_points = []

        # Parcourir les pixels le long de la ligne perpendiculaire (2*nb_pixels + 1)
        for i in range(-nb_pixels, nb_pixels + 1):
            # Calculer le point sur la ligne
            pt = new_base + i * d_perp
            # Arrondir les coordonnées
            x, y = int(round(pt[0])), int(round(pt[1]))

            # Vérifier si le point est dans l'image
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                line_points.append(img[y, x])
            else:
                return np.array([]), False
        
        # Vérifier si tous les points sont nuls
        if np.all(np.array(line_points) == 0):
            return np.array([]), False

        return np.array(line_points), True

    def classify_signal(self, sequence):
        """
        Classe un seul signal selon le nombre et l'ordre des pics.

        - 1 pic → classe 0
        - 2 pics (faible -> fort) → classe 1
        - 2 pics (fort -> faible) → classe 2
        - 3 pics → classe 3

        Args:
            sequence : tableau 1D représentant le signal
            height_threshold : seuil pour détecter les pics

        Returns:
            - classe : classe du signal (0, 1, 2, 3 ou -1 si non classifiable)
        """
        sequence = np.array(sequence)

        # Détection des pics
        peaks, properties = find_peaks(sequence, height=self.height_threshold, distance=3)
        # Récupérer les hauteurs des pics
        heights = properties.get("peak_heights", [])
        
        # Si 1 pic est détecté
        if len(peaks) == 1:
            return 0
        # Si 2 pics sont détectés
        elif len(peaks) == 2:
            # Vérifier si les pics sont dans l'ordre croissant ou décroissant
            return 1 if heights[0] < heights[1] else 2
        # Si 3 pics sont détectés
        elif len(peaks) == 3:
            return 3
        # Si aucun pic ou trop de pics sont détectés
        else:
            return -1

    def predict(self, images):
        """
        Prend une liste de 4 images (2D np.ndarray) en niveaux de gris.
        Retourne la prédiction (classe entre 0 et 3), ou -1 si non classifiable.
        """
        if len(images) != 4:
            raise ValueError("Il faut exactement 4 images.")

        # print(images)
        # print(images[0])
        # print(images[1])
        # print(images[2])
        # print(images[3])

        processed = self.subtract_median(images)
        profile, valid = self.extract_perpendicular_vector(processed)

        if not valid:
            return -1

        return self.classify_signal(profile)
