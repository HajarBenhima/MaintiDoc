"""Garde-fou : refus des demandes d'action sur un équipement.

MaintiDoc est une aide documentaire : il ne pilote aucun équipement. Les demandes
d'action (« Démarre la pompe », « Peux-tu arrêter le moteur ? ») sont refusées
avant tout appel au LLM. Les questions sur une procédure (« Comment démarrer la
pompe ? ») restent autorisées : elles relèvent de la documentation.
"""
import re

# Verbes d'action à l'impératif (2e personne) — début de phrase
_IMPERATIVE = (r"d[ée]marre|red[ée]marre|arr[êe]te|stoppe|lance|allume|[ée]teins|coupe|"
               r"augmente|diminue|baisse|r[ée]duis|r[èe]gle|ouvre|ferme|active|d[ée]sactive|"
               r"pilote|commande|actionne|r[ée]initialise|mets\s+en\s+(?:marche|route)")
# Mêmes verbes à l'infinitif — après « peux-tu », « pouvez-vous », etc.
_INFINITIVE = (r"d[ée]marrer|red[ée]marrer|arr[êe]ter|stopper|lancer|allumer|[ée]teindre|"
               r"couper|augmenter|diminuer|baisser|r[ée]duire|r[ée]gler|ouvrir|fermer|activer|"
               r"d[ée]sactiver|piloter|commander|actionner|r[ée]initialiser|"
               r"mettre\s+en\s+(?:marche|route)")
_ENGLISH = r"start|stop|restart|turn\s+on|turn\s+off|shut\s+down|increase|decrease|open|close|reset"
# Un objet doit suivre le verbe (évite « Coupe A-A : que montre-t-elle ? »)
_OBJECT = (r"\s+(?:(?:la|le|les|un|une|des|du|ce|cet|cette|mon|ma|mes|the|a|an|it|pump|motor)"
           r"(?=\s|$)|l['’])")
_POLITE = r"(?:(?:s['’]il\s+(?:te|vous)\s+pla[îi]t|stp|svp|please|merci\s+de)\s*,?\s*)?"
_MODAL = (r"(?:peux|pourrais|pouvez|pourriez)[\s-]+(?:tu|vous)|tu\s+peux|vous\s+pouvez|"
          r"est[\s-]ce\s+que\s+(?:tu\s+peux|vous\s+pouvez)|can\s+you|could\s+you")

_PATTERNS = [
    re.compile(rf"^{_POLITE}(?:{_IMPERATIVE}|{_ENGLISH}){_OBJECT}", re.IGNORECASE),
    re.compile(rf"^{_POLITE}(?:{_MODAL})\s+(?:{_INFINITIVE}|{_ENGLISH}){_OBJECT}", re.IGNORECASE),
]

REFUSAL = (
    "Je suis un assistant documentaire : je ne peux ni démarrer, ni arrêter, ni régler un "
    "équipement, et je ne remplace pas un technicien habilité. Ces actions doivent être "
    "réalisées sur site, selon les procédures de sécurité en vigueur.\n\n"
    "Je peux en revanche vous indiquer ce que dit la documentation, par exemple la "
    "procédure de démarrage ou les réglages recommandés."
)


def is_action_request(text: str) -> bool:
    """Vrai si une phrase de la demande ordonne d'agir sur un équipement."""
    sentences = re.split(r"[.!?;\n]+", text)
    return any(p.search(s.strip()) for s in sentences for p in _PATTERNS)


def refusal_response() -> dict:
    return {"answer": REFUSAL, "status": "refused", "sources": [], "images": [],
            "tools_used": []}
