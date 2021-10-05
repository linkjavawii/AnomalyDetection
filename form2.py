from flask_wtf import Form,FlaskForm
from wtforms import TextField, SubmitField, TextAreaField
from wtforms.validators import Length, Email, Required
from wtforms import FloatField, validators,StringField,IntegerField,FileField,BooleanField,SelectField
from wtforms.validators import DataRequired, Required
from flask_wtf.file import FileField, FileAllowed, FileRequired

class PointDetectForm(FlaskForm):
        datetime_col = StringField('Datetime column name', validators=[DataRequired()])
        target_col = StringField('Target column name', validators=[DataRequired()])
        lookback = IntegerField(label='Look back', default=10,validators=[Required()])
        epochs = IntegerField(label='Number of epochs', default=5,validators=[Required()])
        firstOutliers = IntegerField(label=' Number of outliers', default=20,validators=[Required()])
        LSTM = BooleanField('LSTM')
        gru = BooleanField('GRU')
        Conv1D = BooleanField('1D-Convolution')
        cnn_lstm = BooleanField('CNN-LSTM')
        BiLstm = BooleanField('Bidirectional-LSTM')
        ARIMA = BooleanField('ARIMA')
        training = SubmitField('Submit')

class SubseqDetectForm(FlaskForm):
        datetime_col = StringField('Datetime column name', validators=[DataRequired()])
        target_col = StringField('Target column name', validators=[DataRequired()])
        subseqlength = IntegerField(label='Subsequence length', default=6,validators=[Required()])
        epochs = IntegerField(label='Number of epochs', default=100,validators=[Required()])
        numberOfNeighbors = IntegerField(label='Number of neighbors', default=100,validators=[Required()])
        numberOfTrees = IntegerField(label='Number of trees', default=500,validators=[Required()])
        firstOutliers = IntegerField(label=' Number of outliers', default=20,validators=[Required()])
        IF = BooleanField('Isolation Forest', default=True)
        pca = BooleanField('Principal Component Analysis')
        knn = BooleanField('KNN Detector')
        LOF = BooleanField('Local Outlier Factor')
        AE = BooleanField('Auto Encoder')
        Conv1D_AE = BooleanField('1D-Convolutional Auto Encoder')
        LSTM_AE = BooleanField('LSTM Auto Encoder')
        kmeans = BooleanField('K-Means (Mahalanobis)')
        training = SubmitField('Submit')
       
class TextDetectForm(FlaskForm):
        target_col = StringField('Text column name', validators=[DataRequired()])
        firstOutliers = IntegerField(label=' Number of outliers', default=20,validators=[Required()])
        epochs = IntegerField(label='Number of epochs', default=30,validators=[Required()])
        numberOfNeighbors = IntegerField(label='Number of neighbors', default=100,validators=[Required()])
        knn = BooleanField('KNN Detector', default=True)
        LOF = BooleanField('Local Outlier Factor')
        LSA = BooleanField('Latent Semantic Analysis')
        AE = BooleanField('Auto Encoder')
        training = SubmitField('Submit')
        
class SupervisedDetectForm(FlaskForm):
        label_col = StringField('Label column for supervised learning', validators=[DataRequired()])
        RF_OS = BooleanField('Random Forest (Over Sampling)', default=True)
        RF_SM = BooleanField('Random Forest (SMOTE)')
        RF_US_ENS = BooleanField('Random Forest (Under Sampling with ensemble methods)')
        XGB_OS = BooleanField('XGBoost (Over Sampling)')
        XGB_SM = BooleanField('XGBoost (SMOTE)')
        XGB_US_ENS = BooleanField('XGBoost (Under Sampling with ensemble methods)')
        num_estimators = IntegerField(label=' Number of Under Sampling', default=600,validators=[Required()])
        training = SubmitField('Submit')

class NumericalDetectForm(FlaskForm):
        firstOutliers = IntegerField(label=' Number of outliers', default=20,validators=[Required()])
        IF = BooleanField('Isolation Forest', default=True)
        knn = BooleanField('KNN Detector')
        LOF = BooleanField('Local Outlier Factor')
        AE = BooleanField('Auto Encoder')
        kmeans = BooleanField('K-Means (Mahalanobis)')
        training = SubmitField('Submit')
        
class MixedDetectForm(FlaskForm):
        epochs = IntegerField(label='Number of epochs', default=100,validators=[Required()])
        numberOfNeighbors = IntegerField(label='Number of neighbors', default=100,validators=[Required()])
        numberOfTrees = IntegerField(label='Number of trees', default=500,validators=[Required()])
        firstOutliers = IntegerField(label=' Number of outliers', default=20,validators=[Required()])
        knn = BooleanField('KNN Detector', default=True)
        LOF = BooleanField('Local Outlier Factor')
        pca = BooleanField('Principal Component Analysis', )
        AE = BooleanField('Auto Encoder')
        IF = BooleanField('Isolation Forest', default=True)
        #kmeans = BooleanField('K-Means (Mahalanobis)')
        training = SubmitField('Submit')


class Choice(FlaskForm):
    point = SubmitField('Change Point Detection in Time Series')
    subseq = SubmitField('Subsequence Detection in Time Series')
    text = SubmitField('Outlier Detection in Text Data')
    supervised = SubmitField('Supervised Outlier Detection')
    #numerical = SubmitField('Outlier Detection in Numerical data')
    mixed = SubmitField('Outlier Detection in Multidimensional data')
        
class UploadForm(FlaskForm):
    file = FileField('CSV file', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'CSV only!')
    ])
