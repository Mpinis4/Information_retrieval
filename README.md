ΟΔΗΓΙΕΣ ΓΙΑ ΤΗΝ ΒΑΣΗ ΔΕΔΟΜΕΝΩΝ
1. Δημιουργούμε μια βάση δεδομένων με όνομα “ Parliament_speeches” στην συνέχεια μπορούμε να εκτελέσουμε τον κώδικα στο αρχείο sql1.sql. ΣΗΜΑΝΤΙΚΟ: στην ακόλουθη γραμμή 
FROM 'C:\temp\Greek_Parliament_Proceedings_1989_2020.csv' 
Διορθώνουμε το path file ώστε να γίνεται η ανάκτηση από το σωστό αρχείο.

Τέλος στην γραμμή:
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:mpampis412@localhost:5432/Parliament_speeches'
αντικαθιστούμε τα ακόλουθα:
postgres=Database Superuser
mpampis=κωδικός του χειριστή του PostgreSQL
5432= Database Port
Parliament_speeches= όνομα της βάσης δεδομένων

ΟΔΗΓΙΕΣ ΓΙΑ ΤΗΝ ΕΝΑΡΞΗ
pip install σε όσες  βιβλιοθήκες δεν διαθέτουμε και στην συνέχεια καλούμαι ένα terminal μέσα στο παραδοτέο φάκελο και τρέχουμε την εντολή “python flask_app.py" στην συνέχεια αφού ξεκινήσει  η εφαρμογή flask ανοίγουμε την διεύθυνση στο browser και εκεί θα βρίσκεται η εφαρμογή μας. Ακολουθεί εικόνα περίπτωσης χρήσης.
