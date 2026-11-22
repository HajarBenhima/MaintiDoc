"""Tests du garde-fou « pas de pilotage d'équipement »."""
import pytest

from maintidoc.safety.guard import is_action_request

ACTIONS = [
    "Démarre la pompe à distance et augmente la pression.",
    "demarre la pompe",
    "Arrête le moteur tout de suite !",
    "Peux-tu démarrer la pompe ?",
    "Pouvez-vous couper l'alimentation du moteur ?",
    "S'il te plaît, ouvre la vanne d'aspiration.",
    "Voici une photo du panneau. Redémarre le compresseur.",
    "Please start the pump.",
    "Mets en marche la pompe 2.",
]

QUESTIONS = [
    "Comment démarrer la pompe ?",
    "Quelle est la procédure pour arrêter le moteur ?",
    "Que faire si la pompe s'arrête ?",
    "Pourquoi la pression augmente-t-elle ?",
    "Que signifie ce voyant orange ? Consulte les dernières interventions et prépare un ticket.",
    "Coupe A-A : que montre-t-elle ?",
    "La pompe triphasée tourne à l'envers, que faire ?",
    "Quelle est la vitesse maximale du fluide dans la tuyauterie d'aspiration ?",
]


@pytest.mark.parametrize("text", ACTIONS)
def test_action_requests_are_refused(text):
    assert is_action_request(text)


@pytest.mark.parametrize("text", QUESTIONS)
def test_documentary_questions_are_allowed(text):
    assert not is_action_request(text)
