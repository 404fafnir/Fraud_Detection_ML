import csv

def div3(SourceFile, TrainFile, VerificationFile, TestFile):
    with open(SourceFile, newline='') as CSVSource:
        spamreader = csv.reader(CSVSource, delimiter=',')
        with open(TrainFile, 'w', newline='') as CSVTrain:
            with open(VerificationFile, 'w', newline='') as CSVVerification:
                with open(TestFile, 'w', newline='') as CSVTest:
                    for row in spamreader:
                        for i in range (3181310):
                            trainwriter = csv.writer(CSVTrain, delimiter=',')
                            trainwriter.writerow(row)
                        for j in range (1590655):
                            verificationwriter = csv.writer(CSVVerification, delimiter=',')
                            verificationwriter.writerow(row)
                        for k in range(1590655):
                            testwriter = csv.writer(CSVTest, delimiter=',')
                            testwriter.writerow(row)


div3('fichier_melange.csv', 'train.csv', 'verif.csv', 'test.csv')                           


