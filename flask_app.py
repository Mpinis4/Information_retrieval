from flask import Flask, render_template, request
from logic import perform_search, extract_keywords, get_member_feature_vectors, cosine_similarity, apply_lsi, apply_kmeans_clustering
from models import db

app = Flask(__name__)

# ρύθμιση της βάσης δεδομένων PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:mpampis@localhost:5432/Parliament_speeches'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
# με την είσοδο στην ιστοσελίδα ο χρήστης βρίσκεται στο index.html
@app.route('/')
def index():
    return render_template('index.html')

#Δρομολόγηση της keyword-extraction.html
@app.route('/keyword-extraction', methods=['GET', 'POST'])
def keyword_extraction():
    if request.method == 'POST':
        name = request.form.get('member_name')  # Λήψη ονόματος μέλους από την είσοδο της φόρμας
        party = request.form.get('political_party')  # Λήψη πολιτικού κόμματος από την είσοδο της φόρμας

        # Συνδυασμός ονόματος και κόμματος αν υπάρχουν
        header = ""
        if name and party:
            header = f"{name} ({party})"
        elif name:
            header = name
        elif party:
            header = party

        # Εξαγωγή λέξεων-κλειδιών ομαδοποιημένων κατά έτος
        keywords_by_year = extract_keywords(name, party)

        if keywords_by_year:
            return render_template('keyword_extraction.html', header=header, keywords_by_year=keywords_by_year)
        else:
            return "Δεν βρέθηκαν ομιλίες για την εξαγωγή λέξεων-κλειδιών"
    
    return render_template('keyword_extraction.html')  # Εμφάνιση φόρμας σε GET

#Δρομολόγηση της topk.html
@app.route('/topk', methods=['GET', 'POST'])
def topk():
    results = None
    k = None

    if request.method == 'POST':
        k = int(request.form.get('k'))

        # Λήψη των χαρακτηριστικών TF-IDF για όλα τα μέλη
        members, vectors = get_member_feature_vectors()

        # Υπολογισμός ομοιότητας συνημιτόνου μεταξύ όλων των ζευγών
        similarity_matrix = cosine_similarity(vectors)

        # Εύρεση των πιο παρόμοιων ζευγαριών μελών
        top_k_pairs = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                similarity_score = similarity_matrix[i][j]
                top_k_pairs.append(((members[i], members[j]), similarity_score))

        # Ταξινόμηση ζευγαριών κατά σκορ ομοιότητας σε φθίνουσα σειρά
        top_k_pairs = sorted(top_k_pairs, key=lambda x: x[1], reverse=True)[:k]

        # Μορφοποίηση των αποτελεσμάτων
        results = [{"pair": f"{pair[0]} και {pair[1]}", "similarity": score} for pair, score in top_k_pairs]

    return render_template('topk.html', results=results, k=k)

#Δρομολόγηση της lsi.html
@app.route('/lsi', methods=['GET', 'POST'])
def lsi_analysis():
    results = None
    n_topics = 5  # Προεπιλεγμένος αριθμός θεμάτων

    if request.method == 'POST':
        n_topics = int(request.form.get('n_topics', 5))  # Λήψη του αριθμού θεμάτων

        # Εφαρμογή LSI και λήψη δομημένων αποτελεσμάτων
        results = apply_lsi(n_topics=n_topics)

        # Ταξινόμηση αποτελεσμάτων κατά κύριο θέμα
        results.sort(key=lambda x: x['main_topic'])

    return render_template('lsi.html', results=results, n_topics=n_topics)

#Δρομολόγηση της clustering.html
@app.route('/clustering', methods=['GET', 'POST'])
def clustering_analysis():
    results = None
    silhouette_avg = None
    n_clusters = 5

    if request.method == 'POST':
        n_clusters = int(request.form.get('n_clusters', 5))  # Λήψη του αριθμού ομάδων από τον χρήστη
        
        #εφαρμογή του clustering
        results, silhouette_avg = apply_kmeans_clustering(n_clusters=n_clusters)

    # άνοιγμα σελίδας clustering.html και αποστολή των αποτελεσμάτων
    return render_template('clustering.html', results=results, silhouette_avg=silhouette_avg, n_clusters=n_clusters)

#Δρομολόγηση της search.html
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_params = {
            "member_name": request.form.get('member_name'),
            "political_party": request.form.get('political_party'),
            "member_region": request.form.get('member_region'),
            "roles": request.form.get('roles'),
            "speech": request.form.get('speech'),
            "starting_date": request.form.get('starting_date'),
            "ending_date": request.form.get('ending_date')
        }
        speakers = perform_search(search_params)
        # άνοιγμα σελίδας search.html και αποστολή των αποτελεσμάτων
        return render_template('search.html', speakers=speakers)
     # άνοιγμα σελίδας search.html και αποστολή κενής λίστας με την είσοδο στην σελίδα
    return render_template('search.html', speakers=[])

if __name__ == '__main__':
    app.run(debug=True)
