#pragma once

#include <fstream>
#include <armadillo>

using namespace std;
using namespace arma;
class HammingNetwork
{
	vector<rowvec> templates;
	rowvec X;
	mat W;
	mat V;
	Col<double> S1;
	Col<double> S2;
	Col<double> Y1;
	Col<double> prevY2;
	Col<double> Y2;

	int k; //количестов эталонов
	int m; //количестов бинарных признаков -размер
	double T;
	double epsilon;
	double error;
		
	HammingNetwork(const char* directoryPath, const char* noisyPath);

	std::vector<double> readFromFile(const char* file);
	void loadTemplates(const char* directoryPath);
	void calculateWeights();
	void recognition();
	Col<double> sumFirstLayer();
	Col<double> sumSecondLayer();
	Col<double> activate(Col<double>);
	double vectorLength(Col<double>);
	void showAnswer();

};

