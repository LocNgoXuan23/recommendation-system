#ifndef RECOMMENDERSYSTEM_H
#define RECOMMENDERSYSTEM_H

#include <QMainWindow>
#include <iostream>
#include <math.h>
#include <fstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include <algorithm>
#include <windows.h>
#include <iomanip>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QMessageBox>

#include "mainwindow.h"

using namespace std;
using namespace Eigen;

#define N 5000   // height of train set
#define H 500    // height of test set
#define M 4      // width of set

static const int width = 60;

class RecommenderSystem
{
protected:
    float Y_data[N][M];         // original data .shape NxM
    float Y_data_T[M][N];       // normalize data. shape NxM
    float Ybar_data[N][M];      // normalize data. shape NxM
    float ratingThreshold;      // rating threshold to recommend
    int n_users;                // number of users
    int n_items;                // number of items
    int K;                      // k neighborhood
    int uuFC;                   // type model
    vector<float> mu;           // average of user columns
    MatrixXd Ybar;              // normalize data. shape n_items x n_users
    MatrixXd S;                 // similarity data

public:
    RecommenderSystem(MatrixXf y_data, int k);
    void fit(Ui::MainWindow *ui);
    float pred(int u, int i, int normalized);
    void print_recommendation(Ui::MainWindow *ui);
    virtual void displayInfo();
    virtual void displayNormalizedMatrix();
    virtual void displaySimilarityMatrix();

private:
    void normalize_Y();
    void similarity(Ui::MainWindow* ui);
    void refresh(Ui::MainWindow* ui);
    float __pred(int u, int i, int normalized = 1);
    vector<float> recommend(int u);
protected:
    float addVector(vector<float> vec);
    float multiplyVectors(vector<float> vec1, vector<float> vec2);
    float cosine_similarity(vector<float> vec1, vector<float> vec2);
    float average(vector<float> vec);
    void displayVector(vector<float> vec);
    void assignMatrix(float A[][M], float B[][M]);
    void assignArray(float A[], float B[], int n);
    void displayArray(int A[], int n);
    void displayMatrix(float matrix[][M]);
    void displayMatrixT(float matrix[][N]);
    int largest(float arr[], int n);
    bool isInVector(vector<float> vec1, int index);
    vector<float> slicing(vector<float> vec, int K);
    vector<float> argsort(vector<float> array);
    vector<float> createZerosVector(vector<float> vec, int n);
};

class UserUserRecommendSystem :public RecommenderSystem {
protected:
public:
    UserUserRecommendSystem(MatrixXf y_data, int k, float rating = 0) :RecommenderSystem(y_data, k) {
        // Y_data
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                Y_data[i][j] = y_data(i, j);
            }
        }

        // Y_data_T
        int i, j;
        for (i = 0; i < M; i++)
            for (j = 0; j < N; j++)
                Y_data_T[i][j] = Y_data[j][i];

        // n_users, n_items
        n_users = largest(Y_data_T[0], N) + 1;
        n_items = largest(Y_data_T[1], N) + 1;

        uuFC = 1;
        ratingThreshold = rating;
    }
    void displayInfo();
    void displayNormalizedMatrix();
    void displaySimilarityMatrix();
};

class ItemItemRecommendSystem :public RecommenderSystem {
public:
    ItemItemRecommendSystem(MatrixXf y_data, int k, float rating = 0) :RecommenderSystem(y_data, k) {
        // Y_data
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                if (j == 0) Y_data[i][j] = y_data(i, 1);
                else if (j == 1) Y_data[i][j] = y_data(i, 0);
                else Y_data[i][j] = y_data(i, j);
            }
        }

        // Y_data_T
        int i, j;
        for (i = 0; i < M; i++)
            for (j = 0; j < N; j++)
                Y_data_T[i][j] = Y_data[j][i];

        // n_users, n_items
        n_users = largest(Y_data_T[0], N) + 1;
        n_items = largest(Y_data_T[1], N) + 1;

        uuFC = 0;
        ratingThreshold = rating;
    }
    void displayInfo();
    void displayNormalizedMatrix();
    void displaySimilarityMatrix();
};

#endif // RECOMMENDERSYSTEM_H
