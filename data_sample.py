import logging
from collections import defaultdict
from pathlib import Path

import utils
from utils import nif_corpus2ner_contexts, iter_corpus

logger = logging.getLogger(__name__)

cls2seed_toks = {
    "Gericht, 2-72-116": ["Asylgerichtshof", "Gerichtshofs", "dem Verwaltungsgericht", "Oberlandesgericht WIEN", "Berufungsgericht", "Bundesverwaltungsgericht"],
    "Behörde, 3-105": ["Standortgemeinde", "LPD Burgenland", "Bundesanstalt", "Autobahnpolizeiinspektion", "Bayerischen Rechtsanwalts- und Steuerberaterversorgung"],
    "Role, Gruppe von Personen, 4-78": ["Patienten", "Angeklagte", "EU-Bürger", "Asylantragstellern", "Todesopfer", "Schutzberechtigte", "Minderheiten-Clans", "Jugendlichen"],
    "Administration, 5": ["Landespolizeidirektion OÖ", "Bayerischen Versicherungskammer-Versorgung", "Staatsanwaltschaft X", "Bundesministerin für Bildung und Frauen", "Gemeindegutsagrargemeinschaft S", "Finanzlandesdirektion für Tirol"],
    # "Bereich, Gebiet, 6": ["Kriegsgebiet", "Königreich", "Baulandbereich", "Heimatort", "Inland", "EU-Gebiet", "Mitgliedstaat"],
    "Land, Staat, 9-12-74-81, 18": ["Republik Slowenien", "Republik Kosovo", "Republik Mazedonien", "Tschetschenien", "Volksrepublik China", "Islamischen Republik Afghanistan", "der Islamischen Republik Iran", "Republik Afghanistan", "Estland", "Bulgarien", "Bundesrepublik Deutschland", "Niederlande", "Marokko"],
    "Information, Quelle, Daten, 20": ["Eurostat-Statistik", "Recherche", "Zivilluftfahrtpersonal-Anweisungen", "Daten", "Aussage", "Berichte", "Faktoren", "Temperaturkontrolllisten", "VwGH-Erkenntnisse"],
    "Verwandtschaft, 51-92": ["Ehegattin", "Verwandte", "Verlobte", "Enkel", "Mutter", "Geschwister", "Cousin", "Kinder"],
    "Straße, Addresse, 60": ["Moskauer Bolotnaya-Platz", "H. Parkplatz", "K.-gasse", "Elisabethallee", "Donau-City-Straße", "Objekt Adresse 1", "U-6 Station Josefstßdter Straße"],
    # "Krankenhaus, Arzt, 75": ["Universitätsklinik für Kinder-und Jugendheilkunde", "Diabetes-Zentrum", "Poliklinik", "Psychiatrie", "Rehaklinik Wien Baumgarten", "Chirurgischen Universitätsklinik", "Krankenhausambulanzen", "Dr. XXXX"],
    "Zocken, Spielhallen, 77": ["Tombolaspiele", "Spielsucht", "Spielhallen", "Lotto", "Kartenspiele", "Spielbetriebes", "Glücksspielautomaten"],
    "Kriegshandlung, Konflikt, 84-115": ["Kampfeinsatz", "Militärdienst", "Angriffen", "Kämpfen in Syrien", "kommunistischen Putsch in Afghanistan", "Ersten Tschetschenienkrieg", "Befreiungskrieg", "Terroranschlag", "Kampf"],
}
seed_tok2cls = {
    x: cl
    for cl, kws in cls2seed_toks.items()
    for x in kws
}


if __name__ == '__main__':
    dataset_folder = Path(utils.config['nif_annotations_folder'])
    logging.basicConfig(level=logging.INFO)

    samples_folder = dataset_folder / 'samples'
    if not samples_folder.exists():
        samples_folder.mkdir()

    nif_corpus = iter_corpus(dataset_folder, doc_limit=None)
    examples = defaultdict(list)
    for eic in nif_corpus2ner_contexts((x[0] for x in nif_corpus), tokens_window=300):
        ner = eic.target_str
        cxt = eic.context
        si, ei = eic.start_ind, eic.end_ind
        ner_cls = eic.tag
        if ner in seed_tok2cls:
            ner_cls = seed_tok2cls[ner]
            if len(examples[ner_cls]) < 100:
                examples[ner_cls].append( [ner, ner_cls, si, ei, cxt] )
                if len(sum(examples.values(), [])) > 500:
                    break
    for cl, instances in examples.items():
        with (dataset_folder / 'samples' / f'{cl}.txt').open('w') as f:
            f.writelines('\t'.join(str(z) for z in x) + '\n' for x in instances)
