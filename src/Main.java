
public class Main {

    public static void main(String[] args) {

        TextCategorization t = new TextCategorization("cheeseDisease.train",
                "cheeseDisease.test");
        t.weightOfWordClassifer = 1.0;
        t.loadTrainingSet();
        t.runTest(true);

    }

}
