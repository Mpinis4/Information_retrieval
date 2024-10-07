import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sqlalchemy import extract
from models import Speaker
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import time

# Φορτωση του μοντέλο NLP για τα ελληνικά
nlp = spacy.load('el_core_news_sm')

#προεπεξεργασίας κειμένου
def preprocess_text(text):
    doc = nlp(text)
    # αφαίρεση των stopwords, σημείων στίξης και αριθμών
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_digit]
    return ' '.join(tokens)


# Ανάκτηση ομιλιών
def fetch_speeches(name, party, year):
    fetch = Speaker.query

    if name:
        fetch = fetch.filter(Speaker.member_name.ilike(f"%{name}%"))
    if year:
        fetch = fetch.filter(extract('year', Speaker.sitting_date) == year)
    if party:
        fetch = fetch.filter(Speaker.political_party.ilike(f"%{party}%"))

    return fetch.limit(100).all()

# υπολογισμός των τιμών TF-IDF
def calculate_tfidf(speeches, top_n=10):
    preprocessed_speeches = [preprocess_text(speech.speech) for speech in speeches]
    # αρχικοποίηση του TF-IDF vectorizer και εφαρμογή του
    vectorizer = TfidfVectorizer(max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_speeches)

    feature_names = vectorizer.get_feature_names_out()

    # Λήψη τιμών TF-IDF
    tfidf_scores = tfidf_matrix.toarray()[0]
    # δείκτες των κορυφαίων N λέξων-κλειδιών
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    #κορυφαίες N λέξεις-κλειδιά
    top_keywords = [feature_names[index] for index in top_indices]

    return top_keywords

# Εξαγωγή κορυφαίων λέξεων-κλειδιών για κάθε έτος από 1989 έως 2020
def extract_keywords(name=None, party=None, top_n=10):

    keywords_by_year = {}

    for year in range(1989, 2021):#λουπα από το 1989 εως 2020
        speeches = fetch_speeches(name, party, year)
        if speeches:
            # Υπολογισμός των κορυφαίων λέξεων-κλειδιών για τις ομιλίες του κάθε τρέχοντος έτους
            keywords = calculate_tfidf(speeches, top_n)
            # Αποθήκευση των λέξεων-κλειδιών για το τρέχον έτος σε πίνακα με ιντεξ το κάθε έτος
            keywords_by_year[year] = keywords

    return keywords_by_year

# υπολογισμός TF-IDF διανυσμάτων για το κάθε μέρος του κοινοβουλίου
def get_member_feature_vectors():
    members = Speaker.query.with_entities(Speaker.member_name).distinct().limit(20).all()  # Ανάκτηση μοναδικών ονομάτων μελών
    member_speeches = {}
    
    for member in members:
        # Ανάκτηση ομιλιών για το εκάστοτε μέλος
        speeches = fetch_speeches(name=member.member_name, party=None, year=None)
        
        # προεπεξεργασία και ένωση όλων των ομιλιών για το εκάστοτε μέλος
        processed_speeches = " ".join([preprocess_text(speech.speech) for speech in speeches])
        member_speeches[member.member_name] = processed_speeches
    
    # υπολογισμός TF-IDF για όλες τις ομιλίες των μελών
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(member_speeches.values())
    
    # επιστροφή λίστας με τα ονόματα των μελών και τον και την τιμή TF-IDF για το κάθε μέλος
    return list(member_speeches.keys()), tfidf_matrix.toarray()

# Υπολογισμός ομοιοτήτων μεταξύ των TF-IDF διανυσμάτων των μελών
def calculate_pairwise_similarities(feature_vectors):
    members = list(feature_vectors.keys())
    vectors = np.array(list(feature_vectors.values()))
    
    # υπολογισμός ομοιότητας συνημιτόνου μεταξύ όλων των ζευγαριών μελών
    similarity_matrix = cosine_similarity(vectors)
    
    # εξαγωγή των τιμών από το πάνω τρίγωνο του πίνακα καθώς αυτός είναι συμμετρικός ως προς την διαγώνιο
    similarities = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            similarity_score = similarity_matrix[i][j]
            similarities.append((members[i], members[j], similarity_score))
    
    return similarities

# Εύρεση των κορυφαίων k πιο όμοιων ζευγών μελών
def find_top_k_similar_members(similarities, k=5):
    
    # ταξινόμηση με βάση την ομοιότηταα σε φθίνουσα σειρά
    sorted_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
    
    return sorted_similarities[:k]

# Κύρια συνάρτηση για την ανάλυση ομοιοτήτων
def analyze_similarities(top_k=10):
    #λήψη των διανυσμάτων χαρακτηριστικών για κάθε μέλος
    feature_vectors = get_member_feature_vectors()
    
    #υπολογισμός συσχετίσεων cosine
    similarities = calculate_pairwise_similarities(feature_vectors)
    
    #εύρεση των κορυφαίων k όμοιων μελών
    top_k_similar_members = find_top_k_similar_members(similarities, top_k)
    #επιστροφή κορυφαίων k όμοιων μελών
    return top_k_similar_members

# LSI για την εύρεση θεματικών περιοχών
def apply_lsi(n_topics=5):
    # ανάκτηση όλων των ομιλιών
    speeches = fetch_speeches(name=None, party=None, year=None)

    #προεπεξεργασία ομιλιών
    processed_speeches = [preprocess_text(speech.speech) for speech in speeches]

    # Δημιουργία CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    term_matrix = vectorizer.fit_transform(processed_speeches)

    # εφαρμογή LSI-SVD
    svd = TruncatedSVD(n_components=n_topics)
    lsi_matrix = svd.fit_transform(term_matrix)

    # εύρεση των λέξεων που συνδέονται με κάθε θέμα
    topic_words = []
    for i in range(n_topics):
        indices = svd.components_[i].argsort()[-10:][::-1]  # 10 κορυφαίες λέξεις
        words = vectorizer.get_feature_names_out()[indices]
        topic_words.append(" ".join(words))  # συμβολοσειρά λέξεων που μας δίνουν την θεματική ενώτητα

    # εξαγωγή του κύριου θέματος για κάθε ομιλία
    main_topics = np.argmax(lsi_matrix, axis=1)

    # δημιουργία εξόδου με την ομιλία, το διάνυσμα και το κύριο θέμα
    output = []
    for idx, speech in enumerate(speeches):
        output.append({
            "speech": speech.speech,
            "vector": lsi_matrix[idx].tolist(),
            "main_topic": topic_words[main_topics[idx]]
        })

    return output

#KMeans clustering
def apply_kmeans_clustering(n_clusters=5):
    #ανάκτηση και προεπεξεργασία ομιλιών
    speeches = fetch_speeches(name=None, party=None, year=None)
    speech_texts = [preprocess_text(speech.speech) for speech in speeches]

    # Μετατροπή των ομιλιών σε διανύσματα με μέγιστο όριο λέξεων 1000
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors = vectorizer.fit_transform(speech_texts)

    # Εφαρμογή KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(vectors)

    # υπολογισμός μέσου όρου του δείκτη σιλουέτας
    silhouette_avg = silhouette_score(vectors, clusters)

    #συνδυασμός ομιλιών και ομάδων που ανήκουν
    results = [{"speech": speech.speech, "cluster": clusters[i]+1} for i, speech in enumerate(speeches)]

    # Ταξινόμηση σύμφωνα με την ομάδα πχ πρωτη θα εμφανίζεται η ομάδα 1 κτλπ
    sorted_results = sorted(results, key=lambda x: x["cluster"])

    return sorted_results, silhouette_avg

# Αναζήτηση
def perform_search(params):
    query = Speaker.query

    #Φιλτράρισμα βάση ημερομηνίας
    if params['starting_date'] and params['ending_date']:
        query = query.filter(Speaker.sitting_date.between(params['starting_date'], params['ending_date']))
    # Φιλτράρισμα βάση ονόματος μέλους
    if params['member_name']:
        query = query.filter(Speaker.member_name.ilike(f"%{params['member_name']}%"))
    # Φιλτράρισμα βάση πολιτικού κόμματος
    if params['political_party']:
        query = query.filter(Speaker.political_party.ilike(f"%{params['political_party']}%"))
    # Φιλτράρισμα βάση περιοχής μέλους
    if params['member_region']:
        query = query.filter(Speaker.member_region.ilike(f"%{params['member_region']}%"))
    # Φιλτράρισμα βάση ρόλου μέλους
    if params['roles']:
        query = query.filter(Speaker.roles.ilike(f"%{params['roles']}%"))
    # Φιλτράρισμα βάση ομιλίας μέλους
    if params['speech']:
        query = query.filter(Speaker.speech.ilike(f"%{params['speech']}%"))
    # Περιορισμός των αποτελεσμάτων σε 100 για να αποικωνίζονται ορθά
    return query.limit(100).all()
