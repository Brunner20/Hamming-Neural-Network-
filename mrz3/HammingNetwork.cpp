#include "HammingNetwork.h"
#include <fstream>
#include <algorithm>


HammingNetwork::HammingNetwork(const char* directoryPath, const char* noisyPath) {
	loadTemplates(directoryPath);
	X = mat(readFromFile(noisyPath));
    X = X.t();

	k = templates.size();
    m = X.size();

    T = m / 2;
    epsilon = 1 / k;
    error = 0.1;

    calculateWeights();
    recognition();

}



void HammingNetwork::recognition(){


    double currentError = 1e15;
    int iteartion = 0;
    bool first = true;

    while (currentError>error)
    {
         S1 = sumFirstLayer();
         Y1 = activate(S1);

         if (first) {
             S2 = S1;
             Y2 = Y1;
             first = false;
             continue;
         }

         S2 = sumSecondLayer();
         prevY2 = Y2;
         Y2 = activate(S2);

         currentError = vectorLength(Y2 - prevY2);
         iteartion++;
         cout << " iteration: " << iteartion << endl;

    }

}

Col<double> HammingNetwork::sumFirstLayer() {
    
    Col<double> sum(k,1);

    for (int i = 0; i < k; i++) {
        double y=0;
        for (int j = 0; j < m; j++) {
            y += W[i, j] * X[i];
        }
        y -= T;
        sum[i, 0] = y;
    
    }
    return sum;
}

Col<double> HammingNetwork::sumSecondLayer() {

    Col<double> sum(k, 1);


    for (int i = 0; i < k; i++) {

        double sumY = 0;;
        for (int j = 0; j < k; j++) {
            if (j != i)
                sumY += Y2[j];
        }
        sum[i, 0] = Y2[i] - epsilon * sumY;

    }
    return sum;
}

Col<double> HammingNetwork::activate(Col<double> toActivate) {
    Col<double> activate;

    for (int i = 0; i < toActivate.n_rows; i++) {
        for (int j = 0; j < toActivate.n_rows; j++) {

            if (toActivate[i, j] <= 0)
                activate[i, j] = 0;
            else if (toActivate[i, j] > 0 && toActivate[i, j] <= T)
                activate[i, j] = toActivate[i, j];
            else if (toActivate[i, j] >= T)
                activate[i, j] = T;

        }
    }
    return activate;
}

double HammingNetwork::vectorLength(Col<double> vec) {

    double sum = 0;
    for (int i = 0; i < vec.size(); i++)
    {
        sum += vec[i] * vec[i];
    }
    return sum;
}

void HammingNetwork::calculateWeights() {


    W = mat(k, m);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            W[i, j] = templates[i][j] / 2;
    
        }
    }

    V = mat(k, k);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {

            if (i == j)
                W[i, j] = 1;
            else
                W[i, j] = -epsilon;
        }
    }


    W.print();
    cout << endl;
    V.print();
}


void HammingNetwork::showAnswer() {

    int number;
    for (int i = 0; i < Y2.n_rows; i++) {
        cout << Y2[i] << endl;
        if (Y2[i] != 0)
            number = i;
    }

    cout << "it is image number: " << number;

}

void HammingNetwork::loadTemplates(const char* directoryPath) {
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(directoryPath)) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string file = ent->d_name;
            if (file.substr(file.find_last_of('.') + 1) == "txt") {
                std::string filePath = directoryPath;
                filePath += "/" + file;
                std::vector<double> tmp = readFromFile(filePath.c_str());
                rowvec pattern(tmp);
                //      pattern=pattern.t();
                templates.push_back(pattern);
            }
        }
        closedir(dir);
    }
    else {
        perror("could not open directory");
        exit(EXIT_FAILURE);
    }
}

std::vector<double> HammingNetwork::readFromFile(const char* file) {
    std::vector<double> vector;
    std::ifstream input(file);
    if (!input.is_open()) {
        std::logic_error wrongFileName("File reading error");
        exit(EXIT_FAILURE);
    }
    else {
        double number;
        while (input >> number) {
            vector.push_back(number);
        }
        input.close();
    }
    return vector;
}
