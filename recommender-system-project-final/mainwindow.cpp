#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "recommendersystem.h"

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

using namespace std;
using namespace Eigen;

#define N 5000   // height of train set
#define H 500    // height of test set
#define M 4      // width of set

MatrixXf Y_train_data(N, M);
MatrixXf Y_test_data(H, M);

UserUserRecommendSystem uuRS = UserUserRecommendSystem(Y_train_data, 30, 2.5);
ItemItemRecommendSystem iiRS = ItemItemRecommendSystem(Y_train_data, 30, 1.5);

// Read data from file url and return data
MatrixXf getData(string fileName, int heightSize, int widthSize) {
    MatrixXf data(heightSize, widthSize);
    int heightCount = 0;

    QFile inputFile(QString::fromStdString(fileName));

    if (inputFile.open(QIODevice::ReadOnly))
    {
       QTextStream in(&inputFile);
       while (!in.atEnd())
       {
          QString line = in.readLine();
          string myText = line.toUtf8().constData();
//          qDebug() << line;
          string s = myText;
          string delim = " ";
          int widthCount = 0;
//          cout << myText << endl;

          auto start = 0U;
          auto end = s.find(delim);
          while (end != string::npos)
          {
              int value = stoi(s.substr(start, end - start));
              //cout << value << endl;
              data(heightCount, widthCount) = value;
              //data[heightCount][widthCount] = value;
              widthCount++;
              start = end + delim.length();
              end = s.find(delim, start);
          }
          int value = stoi(s.substr(start, end));
          data(heightCount, widthCount) = value;
          //data[heightCount][widthCount] = value;
          widthCount++;
          heightCount++;
       }

       inputFile.close();
    }

    return data;
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    Y_train_data = getData("D:\\Code\\Code C++\\Qt creator\\recommender-system-project-final\\some_handled_train_data.txt", N, M);
    Y_test_data = getData("D:\\Code\\Code C++\\Qt creator\\recommender-system-project-final\\some_handled_test_data.txt", H, M);

    connect(ui->userTrainingBtn, SIGNAL(clicked()), this, SLOT(userTraining()));
    connect(ui->itemTrainingBtn, SIGNAL(clicked()), this, SLOT(itemTraining()));
    connect(ui->userCheckPredictBtn, SIGNAL(clicked()), this, SLOT(userCheckPredict()));
    connect(ui->itemCheckPredictBtn, SIGNAL(clicked()), this, SLOT(itemCheckPredict()));
    connect(ui->userDisplayInfoBtn, SIGNAL(clicked()), this, SLOT(userDisplayInfo()));
    connect(ui->itemDisplayInfoBtn, SIGNAL(clicked()), this, SLOT(itemDisplayInfo()));
}

void MainWindow::userTraining()
{
    int k = ui->userKLineEdit->text().toInt();
    float rating = ui->userRatingLineEdit->text().toFloat();

    uuRS = UserUserRecommendSystem(Y_train_data, k, rating);
    uuRS.fit(ui);

    float SE = 0;
        for (int n = 0; n < H; n++) {
            float pred = uuRS.pred(Y_test_data(n, 0), Y_test_data(n, 1), 0);
            SE += pow(pred - Y_test_data(n, 2), 2);
        }
    float RMSE = sqrt(SE / H);
    ui->userRMSELabel->setText(QString("%1").arg(RMSE));

    uuRS.print_recommendation(ui);
    cout << "\tUser-user RecommendSystem, RMSE = " << RMSE << endl;
}

void MainWindow::itemTraining()
{
    int k = ui->itemKLineEdit->text().toInt();
    float rating = ui->itemRatingLineEdit->text().toFloat();

    iiRS = ItemItemRecommendSystem(Y_train_data, k, rating);
    iiRS.fit(ui);

    float SE = 0;
    for (int n = 0; n < H; n++) {
        float pred = iiRS.pred(Y_test_data(n, 0), Y_test_data(n, 1), 0);
        SE += pow(pred - Y_test_data(n, 2), 2);
    }
    float RMSE = sqrt(SE / H);
    ui->itemRMSELabel->setText(QString("%1").arg(RMSE));

    iiRS.print_recommendation(ui);
    cout << "Item-item RecommendSystem, RMSE = " << RMSE;
}

void MainWindow::userCheckPredict() {
    int indexOfUser = ui->userIndexOfUserLineEdit->text().toInt();
    int indexOfItem = ui->userIndexOfItemLineEdit->text().toInt();
    ui->userPredictResultLabel->setText(QString("%1").arg(uuRS.pred(indexOfUser, indexOfItem, 1)));
}

void MainWindow::itemCheckPredict() {
    int indexOfUser = ui->itemIndexOfUserLineEdit->text().toInt();
    int indexOfItem = ui->itemIndexOfItemLineEdit->text().toInt();
    ui->itemPredictResultLabel->setText(QString("%1").arg(iiRS.pred(indexOfUser, indexOfItem, 1)));
}

void MainWindow::userDisplayInfo() {
    uuRS.displayInfo();
    uuRS.displayNormalizedMatrix();
    uuRS.displaySimilarityMatrix();
}

void MainWindow::itemDisplayInfo() {
    iiRS.displayInfo();
    iiRS.displayNormalizedMatrix();
    iiRS.displaySimilarityMatrix();
}

MainWindow::~MainWindow()
{
    delete ui;
}

