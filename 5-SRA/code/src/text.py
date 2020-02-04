'''Objects and methods to support text corpus storage and manipulation'''
import string



def is_alpha(char):
    return char in string.ascii_lowercase

def keep_alphanumeric(doc):
    doc = [char for char in doc]
    out = ''
    for char in doc:
        good = is_alpha(char) or char.isnumeric() or char in [' ', '.']
        if good:
            out += char
        else:
            out += ' '
    return out

def remove_special(docs):
    docs = [doc.lower() for doc in docs]
    docs = [keep_alphanumeric(doc) for doc in docs]
    return docs

# Converts numerals to text, i.e. 1 to 'one'
def remove_numerals(docs):
    docs = [doc.replace('0', ' zero ') for doc in docs]
    docs = [doc.replace('1', ' one ') for doc in docs]
    docs = [doc.replace('2', ' two ') for doc in docs]
    docs = [doc.replace('3', ' three ') for doc in docs]
    docs = [doc.replace('4', ' four ') for doc in docs]
    docs = [doc.replace('5', ' five ') for doc in docs]
    docs = [doc.replace('6', ' six ') for doc in docs]
    docs = [doc.replace('7', ' seven ') for doc in docs]
    docs = [doc.replace('8', ' eight ') for doc in docs]
    docs = [doc.replace('9', ' nine ') for doc in docs]
    return docs

# Removes extra whitespace for a list of text strings. Kludgy, but useful.
def remove_whitespace(docs):
	docs = [' '.join(doc.split()) for doc in docs]
	docs = [doc.rstrip() for doc in docs]
	docs = [doc.lstrip() for doc in docs]
	return docs

# Removes 'nan' strings for a list of strings
def remove_nan(docs):
    docs = [doc.replace('nan', '') for doc in docs]
    return docs

def  expand(doc):
    acr_map = {
        'yom': 'male',
        'yohm': 'male',
        'yof': 'female',
        'yohf': 'female',
        'ym': 'male',
        'yf': 'female',
        'yowm': 'male',
        'yowf': 'female',
        'mvc': 'maximum voluntary contraction',
        'dx': 'diagnosis',
        'pcp': 'personal care physician',
        'abrs': 'acute bacterial rhinosinusitis',
        'nd': 'and',
        'cont': 'contusion',
        'inj': 'injury',
        'fx': 'fracture',
        'fb': 'foreign body',
        'jt': 'joint',
        'co': 'complains of',
        'acc': 'accidentally',
        'bibems': 'brought in by emergency',
        'pt': 'patient',
        'pts': 'patients',
        'lac': 'laceration',
        'loc': 'loss of consciousness',
        'gsw': 'gun shot wound',
        'lt': 'left',
        'rt': 'right',
        'multi': 'multiple',
        'sus': 'sustained',
        'sust': 'sustained',
        'pta': 'prior to arrival',
        'w': 'with',
        'wkr': 'worker',
        'wk': 'work',
        'lbp': 'low blood pressure',
        'spr': 'sprain',
        'bldg': 'building',
        'etoh': 'ethanol',
        'chi': 'closed head injury',
        'x': 'for',
        'iv': 'intravenous',
        'sts': 'states',
        'thru': 'through',
        'tx': 'treatment',
        'tr': 'trauma',
        'ms': 'muscle strain',
        'htn': 'hypertension',
        'pw': 'puncture wound',
        'otj': 'on the job',
        'ed': 'emergency division',
        'puncc': 'puncture',
        'l': 'left',
        'r': 'right',
        'lft': 'left',
        'pn': 'pain',
        'p': 'patient',
        'bk': 'back',
        'dev': 'developed',
        'fel':'fell',
        'mva':'motor vehicle accident',
        'msk':'musculoskeletal pain',
        'str': 'strain',
        'occ':'occupation',
        'expo': 'exposure',
        's': '',
        'y': ''
    }
    sent = ''
    for word in doc:
        sent += acr_map[word] if word in acr_map.keys() else word
        sent+=" "

    return sent

# Function that cleans up a free-text column
def clean_text(docs):

    docs = [doc.lower() for doc in docs]
    docs = remove_special(docs)
    docs = [doc.split() for doc in docs]

    docs = [expand(doc) for doc in docs]

    docs = [doc.replace('f b', ' foreign body ') for doc in docs]
    docs = [doc.replace('yo m', 'male') for doc in docs]
    docs = [doc.replace('y o m', 'male') for doc in docs]
    docs = [doc.replace('yo f', 'female') for doc in docs]
    docs = [doc.replace('y o f', 'female') for doc in docs]
    docs = [doc.replace(' d x ', ' diagnosis ') for doc in docs]
    docs = [doc.replace(' d t ', ' due to ') for doc in docs]
    docs = [doc.replace(' c o ', ' complains of ') for doc in docs]
    docs = [doc.replace(' s p ', ' status post ') for doc in docs]
    docs = [doc.replace('c h i', ' closed head injury') for doc in docs]
    docs = [doc.replace(' 2 2 ', ' secondary to ') for doc in docs]

    docs = remove_whitespace(docs)
    return docs
