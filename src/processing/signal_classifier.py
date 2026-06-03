import numpy as np
import cv2
from scipy.signal import find_peaks

class SignalClassifier():
    def __init__(self, height_threshold=10, subtract_threshold=30, debug=False):
        self.height_threshold = height_threshold
        self.subtract_threshold = subtract_threshold
        self.debug = debug
        self.last_processed = None
        self.last_raw_diff = None  # Diff avant seuillage (pour debug)
        # État détaillé du dernier predict() (pour debug visuel)
        self.last_profile = None
        self.last_peaks = None
        self.last_peak_heights = None
        self.last_center = None
        self.last_tip = None
        self.last_base = None
        self.last_d_perp = None
        self.last_invalid_reason = None

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
        
        # Mediane des 3 premières images (en float pour permettre la diff signée)
        median_image = np.median(np.stack(images[:3]).astype(np.float32), axis=0)

        # Soustraction de la médiane de la 4ème image (différence signée)
        diff = images[3].astype(np.float32) - median_image

        # On garde uniquement les valeurs positives (le balancier "apparaît" sur la frame courante)
        result = np.clip(diff, 0, 255).astype(np.uint8)

        # Recadrage pour obtenir une image carrée
        result = self.crop_to_square(result)
        self.last_raw_diff = result.copy()  # Avant seuillage (pour debug)

        if self.debug:
            print(f"  [CLASSIFIER] diff stats: "
                  f"min={diff.min():.1f} max={diff.max():.1f} "
                  f"mean={diff.mean():.2f} "
                  f"p95={np.percentile(result, 95):.1f} p99={np.percentile(result, 99):.1f} "
                  f"nb_pix>thresh({self.subtract_threshold})={int((result >= self.subtract_threshold).sum())}")

        # Seuillage pour éliminer les pixels faibles
        result[result < self.subtract_threshold] = 0
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

        h, w = img.shape
        self.last_center = np.array([w // 2, h // 2])
        self.last_tip = None
        self.last_base = None
        self.last_d_perp = None

        if len(xs) == 0:
            self.last_invalid_reason = "aucun pixel > 0 dans l'image traitée"
            return np.array([]), False

        coords = np.column_stack((xs, ys))

        center = self.last_center
        # Calculer la distance entre chaque pixel et le centre
        dists = np.linalg.norm(coords - center, axis=1)
        # Trouver le pixel le plus éloigné du centre
        tip = coords[np.argmax(dists)]
        self.last_tip = tip

        vec = tip - center
        norm = np.linalg.norm(vec)

        if norm == 0:
            self.last_invalid_reason = "tip coïncide avec le centre (norm=0)"
            return np.array([]), False

        new_base = center + alpha * vec
        self.last_base = new_base

        # Calculer le vecteur unitaire perpendiculaire au vecteur (centre -> tip)
        d_perp = np.array([-vec[1], vec[0]], dtype=float)
        d_perp = d_perp / norm
        self.last_d_perp = d_perp

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
                # On garde le profil partiel pour debug
                self.last_profile = np.array(line_points)
                self.last_invalid_reason = (
                    f"ligne perpendiculaire sort de l'image au pas i={i} "
                    f"(x={x}, y={y}, taille={w}x{h})"
                )
                return np.array([]), False

        # Vérifier si tous les points sont nuls
        if np.all(np.array(line_points) == 0):
            self.last_profile = np.array(line_points)
            self.last_invalid_reason = "tous les pixels de la ligne sont nuls"
            return np.array([]), False

        self.last_invalid_reason = None
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

        # Mémoriser pour debug
        self.last_peaks = peaks
        self.last_peak_heights = heights

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
        self.last_processed = processed
        # Réinitialiser l'état de debug pour ce predict
        self.last_profile = None
        self.last_peaks = None
        self.last_peak_heights = None
        self.last_invalid_reason = None

        profile, valid = self.extract_perpendicular_vector(processed)
        if valid:
            self.last_profile = profile

        if not valid:
            return -1

        return self.classify_signal(profile)
