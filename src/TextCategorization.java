
/**
 * The Text Recognizer using naive bayes
 *
 */
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;

class TextCategorization {
    // wordCount using word as feature
    ArrayList<HashMap<String, Integer>> wordCounts;
    ArrayList<HashMap<String, Double>> wordVector;

    // alphCount using character i.e. a b c ... as feature
    ArrayList<HashMap<Character, Integer>> alphCounts;
    ArrayList<HashMap<Character, Double>> alphVector;
    String trainingFile;
    String testFile;

    // total probability for alphabet and word
    int cheeseCount, diseaseCount;
    double cheeseProb, diseaseProb;

    int cheeseCountA, diseaseCountA;
    double cheeseProbA, diseaseProbA;

    // the occurrence under this number will be ignore, trying to reduce data noise
    public static final int LOW_OCCUENCE_THRESTHOLD = 0;

    // the weight of word classifier.
    // the classify result will be this number of word classifier and
    // 1 - this number of alphabet classifier
    double weightOfWordClassifer = 0.4;

    public TextCategorization(String trainingFile, String testFile) {
        this.trainingFile = trainingFile;
        this.testFile = testFile;

        this.cheeseCount = 0;
        this.diseaseCount = 0;
        this.cheeseProb = 0;
        this.diseaseProb = 0;

        this.cheeseCountA = 0;
        this.diseaseCountA = 0;
        this.cheeseProbA = 0;
        this.diseaseProbA = 0;

        this.wordCounts = new ArrayList<HashMap<String, Integer>>();
        for (int i = 0; i < 2; i++) {
            this.wordCounts.add(new HashMap<String, Integer>());
        }
        this.wordVector = new ArrayList<HashMap<String, Double>>();
        for (int i = 0; i < 2; i++) {
            this.wordVector.add(new HashMap<String, Double>());
        }
        this.alphCounts = new ArrayList<HashMap<Character, Integer>>();
        for (int i = 0; i < 2; i++) {
            this.alphCounts.add(new HashMap<Character, Integer>());
        }
        this.alphVector = new ArrayList<HashMap<Character, Double>>();
        for (int i = 0; i < 2; i++) {
            this.alphVector.add(new HashMap<Character, Double>());
        }

    }

    /**
     * Load training data set and calculate the vector for naive bayes
     */
    public void loadTrainingSet() {
        try {
            FileReader fr = new FileReader(this.trainingFile);
            BufferedReader br = new BufferedReader(fr);
            String line;

            while ((line = br.readLine()) != null) {
                this.loadDocument(line);
                this.loadDocumentA(line);
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        this.calculateVector();
        this.calculateVectorA();
    }

    /**
     * Run Test on test file report the precision, recall, and F-score
     *
     * @param printDetail
     *            : should we print out the detail of classifying process
     * @return the accuracy of our classifier
     */
    public float runTest(boolean printDetail) {
        int testTotal = 0;
        int correctCount = 0;

        String[] display = { "Cheese", "Disease" };

        int tp = 0, fn = 0, fp = 0, tn = 0;

        try {
            FileReader fr = new FileReader(this.testFile);
            BufferedReader br = new BufferedReader(fr);
            String line;
            if (printDetail) {
                System.out.println(
                        "---------------------------Classify Detail-----------------------");
                System.out.println(String.format("%40s %10s %10s %10s", "Name",
                        "Actual", "Predict", "Result"));
            }

            while ((line = br.readLine()) != null) {
                testTotal++;
                int trueResult = Integer.parseInt(line.substring(0, 1));
                int preditResult = this.classify(line.substring(1));
                if (printDetail) {
                    System.out.println(String.format("%40s %10s %10s %10s",
                            line.substring(1).trim(), display[trueResult - 1],
                            display[preditResult - 1],
                            trueResult == preditResult));
                }

                // true positive
                if (trueResult == 2 && preditResult == 2) {
                    tp++;
                }

                // false negative
                if (trueResult == 2 && preditResult == 1) {
                    fn++;
                }

                // false positive
                if (trueResult == 1 && preditResult == 2) {
                    fp++;
                }

                // true negative
                if (trueResult == 1 && preditResult == 1) {
                    tn++;
                }

                if (trueResult == preditResult) {
                    correctCount++;
                }
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println();
        System.out.println("---------Matrix--------");

        System.out.println(String.format("%14s %7s %8s", "Actual\\Predict",
                "|Disease", "|Cheese |"));

        System.out.println(String.format("%14s |%7d |%7d|", "Disease", tp, fn));
        System.out.println(String.format("%14s |%7d |%7d|", "Cheese", fp, tn));
        System.out.println("---------Status--------");
        System.out.println("Precision:" + ((float) tp / (tp + fp)));
        System.out.println("Recall:" + ((float) tp / (tp + fn)));
        System.out.println("F-score:" + ((float) 2 * tp / (2 * tp + fn + fp)));
        System.out.println("Accuracy:" + (float) correctCount / testTotal);
        this.dumpStatus();
        System.out.println("INFO:");
        System.out.println(this.help());
        return (float) correctCount / testTotal;
    }

    /**
     * put string data into word counter arrayList
     *
     * @param doc
     *            the line that contains feature
     */
    private void loadDocument(String doc) {
        doc = filter(doc);
        String[] tokens = doc.split("\\s+");
        // The first token is the class id
        int classId = Integer.parseInt(tokens[0]);
        classId--;

        // iterate through the rest of the tokens
        for (int i = 1; i < tokens.length; i++) {
            if (classId == 0) {
                this.cheeseCount++;
            } else {
                this.diseaseCount++;
            }
            if (!this.wordCounts.get(classId).containsKey(tokens[i])) {
                this.wordCounts.get(classId).put(tokens[i], 1);
            } else {
                int count = this.wordCounts.get(classId).get(tokens[i]);
                count++;
                this.wordCounts.get(classId).put(tokens[i], count);

            }
        }

    }

    /**
     * Load data into alphabet counter arrayList
     *
     * @param doc
     *            the line that contain feature
     */
    private void loadDocumentA(String doc) {
        doc = filter(doc);

        String[] tokens = doc.split("\\s+");
        // The first token is the class id
        int classId = Integer.parseInt(tokens[0]);
        classId--;

        // iterate through the rest of the tokens
        for (int i = 1; i < tokens.length; i++) {
            if (classId == 0) {
                this.cheeseCountA++;
            } else {
                this.diseaseCountA++;
            }
            // for each character, we try to add to our array list
            for (Character k : tokens[i].toCharArray()) {
                k = Character.toLowerCase(k);
                if (this.isCharacter(k)) {
                    if (!this.alphCounts.get(classId).containsKey(k)) {
                        this.alphCounts.get(classId).put(k, 1);
                    } else {
                        int count = this.alphCounts.get(classId).get(k);
                        count++;
                        this.alphCounts.get(classId).put(k, count);

                    }
                }
            }
        }

    }

    /**
     * Test if a character is alphabetical a-zA-Z
     *
     * @param k
     *            the character to test
     * @return
     */
    private boolean isCharacter(char k) {
        int i = k;

        return i >= 97 && i <= 122;
    }

    /**
     * the classify method.
     *
     * @param word
     *            the word to classify
     * @return the label of this word
     */
    public int classify(String word) {

        word = filter(word);
        String[] tokens = word.split("\\s+");

        double cheesePA = this.cheeseProbA;

        double diseasePA = this.diseaseProbA;

        for (String key : tokens) {
            for (Character token : key.toCharArray()) {
                token = Character.toLowerCase(token);
                if (this.isCharacter(token)) {
                    if (this.alphVector.get(0).containsKey(token)) {
                        cheesePA *= this.alphVector.get(0).get(token);
                    }
                    if (this.alphVector.get(1).containsKey(token)) {
                        diseasePA *= this.alphVector.get(1).get(token);
                    }
                }
            }
        }

        double cheeseP = this.cheeseProb;

        double diseaseP = this.diseaseProb;

        for (String token : tokens) {
            if (this.wordVector.get(0).containsKey(token)) {
                cheeseP *= this.wordVector.get(0).get(token);
            }
            if (this.wordVector.get(1).containsKey(token)) {
                diseaseP *= this.wordVector.get(1).get(token);
            }

        }
        double finalCheese = cheeseP * this.weightOfWordClassifer
                + cheesePA * (1 - this.weightOfWordClassifer);
        double finalDisease = diseaseP * this.weightOfWordClassifer
                + diseasePA * (1 - this.weightOfWordClassifer);
        return finalCheese > finalDisease ? 1 : 2;

    }

    /**
     * calculate the word vector using Laplace smoothing
     */
    private void calculateVector() {
        this.cheeseProb = (float) this.cheeseCount
                / (this.cheeseCount + this.diseaseCount);
        this.diseaseProb = (float) this.diseaseCount
                / (this.cheeseCount + this.diseaseCount);

        int NumOfVector = this.wordCounts.get(0).size();

        for (String key : this.wordCounts.get(1).keySet()) {
            if (!this.wordCounts.get(0).containsKey(key)) {
                NumOfVector++;
            }
        }

        int[] wordInTotal = { 0, 0 };
        for (int i = 0; i < 2; i++) {
            HashMap<String, Integer> dic = this.wordCounts.get(i);
            for (int count : dic.values()) {
                wordInTotal[i] += count;
            }
        }
        for (int i = 0; i < 2; i++) {
            HashMap<String, Integer> dic = this.wordCounts.get(i);
            for (String key : dic.keySet()) {
                if (dic.get(key) >= LOW_OCCUENCE_THRESTHOLD) {

                    double prob = this.kernel((double) (dic.get(key) + 1)
                            / (wordInTotal[i] + NumOfVector));
                    this.wordVector.get(i).put(key, prob);
                    if (!this.wordCounts.get(1 - i).containsKey(key)) {
                        prob = this.kernel((double) (1)
                                / (wordInTotal[1 - i] + NumOfVector));
                        this.wordVector.get(1 - i).put(key, prob);
                    }
                }
            }
        }
    }

    /**
     * calculate the alphabet vector with Laplace smoothing
     */
    private void calculateVectorA() {
        this.cheeseProbA = (float) this.cheeseCountA
                / (this.cheeseCountA + this.diseaseCountA);
        this.diseaseProbA = (float) this.diseaseCountA
                / (this.cheeseCountA + this.diseaseCountA);

        int NumOfVector = this.alphCounts.get(0).size();

        for (Character key : this.alphCounts.get(1).keySet()) {
            if (!this.alphCounts.get(0).containsKey(key)) {
                NumOfVector++;
            }
        }

        int[] wordInTotal = { 0, 0 };
        for (int i = 0; i < 2; i++) {
            HashMap<Character, Integer> dic = this.alphCounts.get(i);
            for (int count : dic.values()) {
                wordInTotal[i] += count;
            }
        }
        for (int i = 0; i < 2; i++) {
            HashMap<Character, Integer> dic = this.alphCounts.get(i);
            for (Character key : dic.keySet()) {
                if (dic.get(key) >= LOW_OCCUENCE_THRESTHOLD) {

                    double prob = this.kernel((double) (dic.get(key) + 1)
                            / (wordInTotal[i] + NumOfVector));
                    this.alphVector.get(i).put(key, prob);
                    if (!this.alphCounts.get(1 - i).containsKey(key)) {
                        prob = this.kernel((double) (1)
                                / (wordInTotal[1 - i] + NumOfVector));
                        this.alphVector.get(1 - i).put(key, prob);
                    }
                }
            }
        }
    }

    /**
     * This method is abandoned, used to adjust the probability output to
     * Eliminated the noise
     *
     * @param d
     *            input probability
     * @return output probability
     */
    private double kernel(double d) {

        return d;
    }

    /**
     * print out the status of vector and counter
     */
    public void dumpStatus() {
        int NumOfVector = this.wordCounts.get(0).size();

        for (String key : this.wordCounts.get(1).keySet()) {
            if (!this.wordCounts.get(0).containsKey(key)) {
                NumOfVector++;
            }
        }

        int[] wordInTotal = { 0, 0 };

        for (int i = 0; i < 2; i++) {
            HashMap<String, Integer> dic = this.wordCounts.get(i);
            for (int count : dic.values()) {
                wordInTotal[i] += count;
            }
        }
        System.out.println("---------Status----------");
        System.out.println("word in Cheese  : " + wordInTotal[0]);
        System.out.println("word in Disease  : " + wordInTotal[1]);
        System.out.println("Number of vector : " + NumOfVector);

        NumOfVector = this.alphCounts.get(0).size();

        for (Character key : this.alphCounts.get(1).keySet()) {
            if (!this.alphCounts.get(0).containsKey(key)) {
                NumOfVector++;
            }
        }
        System.out.println();
        wordInTotal[0] = 0;
        wordInTotal[1] = 0;

        for (int i = 0; i < 2; i++) {
            HashMap<Character, Integer> dic = this.alphCounts.get(i);
            for (int count : dic.values()) {
                wordInTotal[i] += count;
            }
        }
        System.out.println("Alphabet in Cheese  : " + wordInTotal[0]);
        System.out.println("Alphabet in Disease  : " + wordInTotal[1]);
        System.out.println("Number of vector : " + NumOfVector);
        System.out.println("----------End------------");
    }

    public void dumpVector() {

        System.out.println("---------Cheese-Word------");
        System.out.println(this.wordCounts.get(0));
        System.out.println(this.wordVector.get(0));
        System.out.println("---------Disease-Word-----");

        System.out.println(this.wordCounts.get(1));
        System.out.println(this.wordVector.get(1));
        System.out.println("----------End------------");

        System.out.println("---------Cheese-Alph-----");
        System.out.println(this.alphCounts.get(0));
        System.out.println(this.alphVector.get(0));
        System.out.println("---------Disease-Alph----");

        System.out.println(this.alphCounts.get(1));
        System.out.println(this.alphVector.get(1));
        System.out.println("----------End------------");
    }

    /**
     * Get rid of parenthesis and '.' ','
     *
     * @param str
     *            string to purify
     * @return purified string
     */
    public static String filter(String str) {
        str = str.replaceAll("\\(", "");
        str = str.replaceAll("\\)", "");
        str = str.replaceAll("\\.", "");
        str = str.replaceAll("\\,", "");
        return str;
    }

    @Override
    public String toString() {
        return "Cheese \n" + this.wordVector.get(0).toString()
                + "\n\n Disease \n" + this.wordVector.get(1).toString();
    }

    /**
     * Return the info of how to calculate the status
     *
     * @return
     */
    public String help() {
        String info = "Accuray = (tp+tn) / (tp+fn+fp+tn) \n";
        info += "Precision = tp / (tp+fn)    PS: biased towards C(Disease|Disease) & C(Disease|Cheese)\n";
        info += "Recall = tp / (tp + tn)     PS: biased towards C(Disease|Disease) & C(Cheese|Disease)\n";
        info += "F-socre = 2tp / (2tp+tn+fn) PS: biased towards all except C(Cheese|Cheese)\n";
        return info;

    }

}
