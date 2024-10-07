from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Speaker(db.Model):
    __tablename__ = 'Speaker'
    id = db.Column(db.Integer, primary_key=True)
    member_name = db.Column(db.String(150))
    sitting_date = db.Column(db.Date)    
    parliamentary_period = db.Column(db.String(150))
    parliamentary_session = db.Column(db.String(150))
    parliamentary_sitting = db.Column(db.String(150))
    political_party = db.Column(db.String(150))
    government = db.Column(db.String(250))
    member_region = db.Column(db.String(150))
    roles = db.Column(db.String(250))
    member_gender = db.Column(db.String(10))
    speech = db.Column(db.Text)

