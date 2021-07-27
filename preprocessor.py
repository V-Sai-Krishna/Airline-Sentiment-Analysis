from util import *

def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

class ProcessText():
  def __init__(self):
  	super(ProcessText, self).__init__()
  	self.sym_spell = pickle.load(open('symspell.pkl', 'rb'))
  	self.words = open("words-by-frequency.txt").read().split()
  	self.wordcost = pickle.load(open('wordcost.pkl', 'rb'))
  	self.maxword = pickle.load(open('maxword .pkl', 'rb'))

  def infer_spaces(self,s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-self.maxword):i]))
        return min((c + self.wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return list(reversed(out))

  def SentenceSegmentation(self, text):
    tokenizer_punkt = nltk.data.load('tokenizers/punkt/english.pickle')
    segmentedText = tokenizer_punkt.tokenize(text.strip())
    return segmentedText

  def Tokenization(self, text):
      tokenizedText = []
      # APPOSTOPHES = {"'s" : "is", "'re" : "are", "'t": " not", "'ve": "have", "cause": "because", "'d": "would", "'ll": "will"} 
      p_t = nltk.tokenize.treebank.TreebankWordTokenizer()
      for i in text:
          text_raw_token = p_t.tokenize(i)
          for j in text_raw_token :
              if (j == "") or (j in string.punctuation):
                  continue
              tokenizedText.append(j)
      return tokenizedText
  
  def StopwordRemoval(self, text):
    stopwordRemovedText = None
    stop_words = set(stopwords.words('english')) 
    stopwordRemovedText = [j for j in text if not j in stop_words and len(j)>2]
    if(len(stopwordRemovedText)>0):
      return stopwordRemovedText
    else:
      return text

  def InflectionReduction(self,text):
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    reducedText = [lem.lemmatize(lem.lemmatize(j,'n'),'v') for j in text]
    return reducedText

  def process(self,texts):
    ptw=[]
    pts=[]
    for text in texts:
      emojiless=convert_emojis(text) #Converts Emojis into text
      linkless=re.sub(r'http\S+', '', emojiless) #Removes links
      d=re.sub(r'@\S+','',linkless) #Removes name
      # d = str(sym_spell.lookup_compound(d, max_edit_distance=2)[0])[:-6] # Spelling Correction
      d=self.SentenceSegmentation(str(d).lower()) #Sentence Segmentation
      for i in range(len(d)):
        for char in d[i]:
          if char =="_":
            d[i]=d[i].replace(char,' ')
          elif not char.isalpha():
            d[i]=d[i].replace(char,' ')
          else:
            d[i]=d[i]
      d1=self.Tokenization(d)
      d2=[]
      for i in range(len(d1)):
        d2=d2+(self.infer_spaces(d1[i]))
      for i in range(len(d2)):
        d2[i]=str(self.sym_spell.lookup(d2[i], Verbosity.CLOSEST, max_edit_distance=5,include_unknown=True)[0]).split(",")[0]
      d2=self.StopwordRemoval(d2)
      d3=self.InflectionReduction(d2)
      d4=' '.join(d3)
      ptw.append(d3)
      pts.append(d4)
    return ptw,pts
