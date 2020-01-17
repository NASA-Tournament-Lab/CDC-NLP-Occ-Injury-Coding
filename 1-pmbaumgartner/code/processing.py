import re

import pkg_resources

from symspellpy import SymSpell

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
)

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

capture = re.compile(r"\d{1,2}\s?Y\s?O?\s?[WMF]{0,2}")

benchmark_replacements = [
    ("dx", "diagnosis"),
    ("d x", "diagnosis"),
    ("c o", "complains of"),
    ("bibems", "brought in by ems"),
    ("pt", "patient"),
    ("pts", "patients"),
    ("lac", "laceration"),
    ("lt", "left"),
    ("rt", "right"),
    ("sus", "sustained"),
    ("fx", "fracture"),
    ("bldg", "bleeding"),
    ("s p", "status post"),
    ("p w", "patient with"),  # not original
    ("w", "with"),
    ("gsw", "gun shot wound"),
    ("etoh", "ethanol"),
    ("loc", "loss of consciousness"),
    ("pta", "prior to arrival"),
    ("x", "for"),
    ("chi", "closed head injury"),
    ("2 2", "secondary to"),
    ("lbp", "low blood pressure"),
    ("htn", "hypertension"),
    ("pw", "puncture wound"),
]

custom_repl = [
    ("fb", "foreign body"),
    ("sob", "shortness of breath"),
    ("acc", "accidentally"),
    ("int he", "in the"),
    ("freq", "frequent"),
    ("alaceration", "a laceration"),
    ("r", "right"),
    ("l", "left"),
    ("ed", "emergency department"),
    ("edp", "emergency department patient"),
    ("c h i", "closed head injury"),
    ("inj", "injury"),
    ("sust", "sustained"),
    ("ytd", "yesterday"),
    ("yest", "yesterday"),
    ("mva", "motor vehicle accident"),
    ("mvc", "motor vehicle collision"),
    ("mo", "month"),
    ("inrt", "in right"),
    ("inlt", "in left"),
    ("cont", "contusion"),
    ("conts", "contusions"),
    ("exp", "exposure"),
    ("shlder", "shoulder"),
    ("sded", "sided"),
    ("sd", "side"),
    ("nk", "neck"),
    ("bk", "back"),
    ("cohb", "carboxyhemoglobin"),
    ("abra", "abrasion"),
    ("abr", "abrasion"),
    ("abras", "abrasions"),
    ("abrs", "abrasions"),
    ("sev", "several"),
    ("xs", "times"),
    ("multi?", "multiple"),
    ("fxs", "fractures"),
    ("sah", "subarachnoid hemorrhage"),
    ("sdh", "subdural hemorrhage"),
    ("tbi", "traumatic brain injury"),
    ("tr", "trauma"),
    ("sts", "states"),
    ("pc", "piece"),
    ("dev", "developed"),
    ("cwp", "chronic widespread pain"),
    ("bld", "blood"),
    ("abd", "abdomen"),
    ("occ expo", "occupational exposure"),
    ("occ", "occupation"),
    ("expo", "exposure"),
    ("foosh", "fallen onto an outstretched hand"),
    ("foosa", "fallen onto an outstretched arm"),
    ("ms", "muscle"),
    ("imm", "immediate"),
    ("tdy", "today"),
    ("perp", "perpetrator"),
    ("eval", "evaluation"),
    ("fa", "forearm"),
    ("astham", "asthma"),
    ("h o", "history of"),
    ("derm", "dermatitis"),
    ("accid", "accidentally"),
    ("unk", "unknown"),
    ("onrt", "on right"),
    ("onlt", "on left"),
    ("spr", "sprain"),
    ("glf", "ground level fall"),
    ("er", "emergency room"),
    ("fr", "from"),
    ("1day", "1 day"),
    ("msk", "musculoskeletal injury"),
    ("mcp", "metacarpophalangeal joint"),
    ("fib", "fibula"),
    ("tib", "tibia"),
    ("pn", "pain"),
    ("hx", "history"),
    ("cp", "chest pain"),
    ("cwp", "chest wall pain"),
    ("fing", "finger"),
    ("otj", "on the job"),
    ("fbs", "foreign bodies"),
    ("naus", "nausea"),
    ("vom", "vomiting"),
    ("rts", "reports"),
    ("tx", "treating"),
    ("med", "medical"),
    ("bil", "bilateral"),
    ("bck", "back"),
    ("diff", "difficulty"),  # also "different" sometimes
    ("amb", "ambulate"),
    ("wt", "weight"),
    ("thur", "through"),
    ("wc", "workers compensation"),
    ("s d f", "slip and fell"),
    ("t d f", "trip and fell"),
    ("bwd?", "backward"),
    ("lh d", "lightheaded and dizzy"),
    ("ld", "lightheaded"),
    ("mv", "motor vehicle"),
    ("elevated bp", "elevated blood pressure"),
    ("bp", "back pain"),
    ("str", "strain"),
    ("px", "pain"),
    ("ct", "contusion"),
    ("nh", "nursing home"),
    ("f b", "foreign body"),
    ("poss", "possible"),
    ("uv", "ultraviolet"),
    ("rad", "radiating"),  # sometimes 'radiation'
    ("int", "intermittent"),
    ("h a", "headache"),
    ("ha", "headache"),
    ("c a", "from"),  # Not sure, makes grammatical sense
    ("co pain", "complains of pain"),
    ("fwy", "freeway"),
    ("mcc", "multiple car collision"),
    ("sb", "seatbelt"),
    ("sb ab", "seatbelt airbag"),
    ("ab sb", "airbag seatbelt"),
    ("ab", "abrasion"),
    ("p t", "patient"),
    ("contu", "contusion"),
    ("s b", "seatbelt"),
    ("lh", "lightheadedness"),
    ("dness", "dizziness"),
    ("n v", "nausea vomiting"),
    ("fwd?", "forward"),
    ("hosp", "hospital"),
    ("b u", "break up"),
    ("b t", "between"),
    ("lwbs", "left without being seen"),
    ("ama", "against medical advice"),
    ("ind", "index"),
    ("mid", "middle"),
    ("f arm", "forearm"),
    ("w o", "without"),
    ("with o", "without"),
    ("b c", "because"),
    ("lwot", "left without treatment"),
    ("sth", "something"),
    ("mech", "mechanical"),
    ("wkr", "worker"),
    ("sof", "symptoms of"),
    # Cutoff here for models submitted oct 25
    ("o t", "owing to"),
    ("d t", "due to"),
    ("s d", "slipped"),
    ("hi", "head injury"),
    ("in or", "in operating room"),
    ("the or", "the operating room"),
    ("to or", "to operating room"),
    ("sx", "surgery"),
    ("fd", "fire department"),
    ("ns", "not sure"),
    ("ls", "lumbosacral"),
    ("po", "police officer"),
    ("md", "doctor"),
    ("hd", "head"),
    ("ff", "firefighter"),
    ("cna", "nurse assistant"),
    ("obj", "object"),
    ("inv", "involved"),
    ("ext", "extremity"),
    ("ona", "on a"),
    ("mth", "month"),
    ("rd", "rider"),
    ("pd", "police"),
    ("rn", "nurse"),
    ("lacs", "lacerations"),
    ("lll", "left lower leg"),
    ("rll", "right lower leg"),
    ("thru", "through"),
    ("biba", "brought in by ambulance"),
    ("exac", "exacerbation"),
    ("yo", "year old"),
    ("conj", "conjunctivitis"),
    ("pca", "personal care assistant"),
    ("rlq", "right lower quadrant"),
    ("llq", "left lower quadrant"),
    ("ca", "car accident"),
    ("chst", "chest"),
    ("ba ck", "back"),
    ("lg", "large"),
    ("sm", "small"),
    ("rif", "right index finger"),
    ("lif", "left index finger"),
    ("rmf", "right middle finger"),
    ("lmf", "left middle finger"),
]

spelling_corrections = [
    ("heachache", "headache"),
    ("nau", "nausea"),
    ("vommiting", "vomiting"),
    ("palpatations", "palpitations"),
    ("coug", "cough"),
    ("headace", "headache"),
    ("physicial", "physical"),
    ("nausa", "nausea"),
    ("headach", "headache"),
    ("migrane", "migraine"),
    ("breating", "breathing"),
    ("cought", "coughing"),
    ("assult", "assault"),
    ("couhging", "coughing"),
    ("naus", "nausea"),
    ("assualted", "assault"),
    ("nauseaus", "nausea"),
    ("ashtma", "asthma"),
    ("pian", "pain"),
    ("vomting", "vomiting"),
    ("hemmorrhage", "hemorrhage"),
    ("nausia", "nausea"),
    ("astma", "asthma"),
    ("pregant", "pregnant"),
    ("assualt", "assault"),
    ("sypmtoms", "symptoms"),
    ("nauses", "nausea"),
    ("whelps", "welts"),
    ("vomitting", "vomiting"),
    ("hypertens", "hypertension"),
    ("vomitng", "vomiting"),
    ("breth", "breath"),
    ("dizz", "dizzy"),
    ("bodyaches", "body ache"),
    ("asthm", "asthma"),
    ("asthama", "asthma"),
    ("tetnus", "tetanus"),
    ("voming", "vomiting"),
    ("nausae", "nausea"),
    ("vomitted", "vomiting"),
    ("dizzyness", "dizziness"),
    ("nuasea", "nausea"),
    ("nasuea", "nausea"),
    ("nausated", "nausea"),
    ("rigth", "right"),
    ("dizzines", "dizziness"),
    ("astham", "asthma"),
    ("tramatic", "traumatic"),
    ("syptoms", "symptoms"),
    ("headahce", "headache"),
    ("reation", "reaction"),
    ("diarrhe", "diarrhea"),
    ("pneumo", "pneumonia"),
    ("asth", "asthma"),
    ("abcess", "abscess"),
    ("symptons", "symptoms"),
    ("nauseas", "nausea"),
    ("headahe", "headache"),
    ("diziness", "dizziness"),
    ("dirrhea", "diarrhea"),
    ("assulted", "assault"),
    ("electricution", "electrocution"),
    ("couging", "coughing"),
    ("vomtiing", "vomiting"),
    ("respitory", "respiratory"),
]

# c t
# ATR
# AOF :area of focus?
# AMB
# ANT
# C (causing, citing?)

date_repl = (
    [("1wk", "1 week"), ("1week", "1 week"), ("1wa", "1 week ago")]
    + [("(\d{1,2})\s?we{0,2}ks?", r"\1 weeks"), ("(\d{1,2})wag?o?", r"\1 weeks ago")]
    + [("wk", "work"), ("wks", "works")]  # important this is last
    + [("(\d{1,2})d", r"\1 days "), ("(\d{1,2})dag?o?", r"\1 days ago ")]
    + [("1hr", "1 hour "), ("(\d{1,2})hrs?", r"\1 hours "), ("hr", "hour")]
    + [("1ma", "1 month ago "), ("(\d{1,2})mag?o?", r"\1 hours ")]
)


height_repl = (
    [("1fth", "1 foot high"), ("1ft", "1 foot")]
    + [("(\d{1,2})fth", r"\1 feet high"), ("(\d{1,2})\s?ft", r"\1 feet")]
    + [("ft", "foot")]  # important this is last
)


ordinal_repl = [
    ("1st", "first"),
    ("2nd", "second"),
    ("3rd", "third"),
    ("4th", "fourth"),
    ("5th", "fifth"),
    ("6th", "sixth"),
    ("7th", "seventh"),
    ("8th", "eighth"),
    ("9th", "ninth"),
    ("10th", "tenth"),
    ("11th", "eleventh"),
    ("12th", "twelfth"),
    ("13th", "thirteenth"),
    ("14th", "fourteenth"),
    ("15th", "fifteenth"),
    ("16th", "sixteenth"),
    ("17th", "seventeenth"),
    ("18th", "eighteenth"),
    ("19th", "nineteenth"),
    ("20th", "twenteith"),
]

dow_repl_base = [
    ("sun", "sunday"),
    ("mon", "monday"),
    ("tues", "tuesday"),
    ("wed", "wednesday"),
    ("thurs", "thursday"),
    ("fri", "friday"),
    ("sat", "saturday"),
]

# adds "on" or "last" to day of week
dow_repl_context = []
for k, v in dow_repl_base:
    dow_repl_context.append(("on " + k, v))
    dow_repl_context.append(("last " + k, v))
    dow_repl_context.append(("since " + k, v))
    if k not in ("sat", "sun"):
        dow_repl_context.append((k, v))


am_pm = [
    ("this am", "this morning"),
    ("in am", "in morning"),
    ("last am", "last morning"),
    ("this pm", "this evening"),
    ("in pm", "in evening"),
    ("last pm", "last evening"),
]


replacements = (
    benchmark_replacements
    + custom_repl
    + spelling_corrections
    + date_repl
    + height_repl
    + ordinal_repl
    + dow_repl_context
    + am_pm
)


replacement_regex = []
for original, replacement in replacements:
    original_regex = re.compile(r"\b{}\b".format(original))
    replacement_regex.append((original_regex, replacement))

capture_male_lower = re.compile(r"(\d{1,2})\s?y\s?o?\s?\S?(male|m)")
capture_female_lower = re.compile(r"(\d{1,2})\s?y\s?o?\s?\S?(female|f)")


def preprocessing_bert(text):
    clean_text = text.strip().lower()

    clean_text = re.sub("dx", " dx ", clean_text)
    clean_text = re.sub("hx", " hx ", clean_text)
    clean_text = re.sub("fx", " fx ", clean_text)

    clean_text = " ".join(clean_text.split())

    clean_text = re.sub(capture_male_lower, r"\1 year old male", clean_text)
    clean_text = re.sub(capture_female_lower, r"\1 year old female", clean_text)

    clean_text = re.sub(r"x(\d{1,2})", r"for \1", clean_text)
    clean_text = re.sub(r"\b(pt)(\S+?)\b", r"patient \2", clean_text)

    for repl_value, repl_with in replacement_regex:
        clean_text = re.sub(repl_value, repl_with, clean_text)

    clean_text = re.sub(r" diagnosis ", ". diagnosis ", clean_text)

    if clean_text[-1] != ".":
        clean_text = clean_text + "."

    return clean_text


def preprocess_symspell(text):
    btext = preprocessing_bert(text)
    suggestions = sym_spell.lookup_compound(
        btext.lower(), max_edit_distance=2, ignore_non_words=True
    )
    result = suggestions[0].term.strip()
    result = re.sub(r" diagnosis ", ". diagnosis ", result)
    if result[-1] != ".":
        result = result + "."
    return result
