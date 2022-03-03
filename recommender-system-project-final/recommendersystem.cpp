#include "recommendersystem.h"
#include "ui_mainwindow.h"

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
#include <QString>

using namespace std;
using namespace Eigen;

#define N 5000   // height of train set
#define H 500    // height of test set
#define M 4      // width of set

RecommenderSystem::RecommenderSystem(MatrixXf y_data, int k)
{
    K = k;
}

void RecommenderSystem::fit(Ui::MainWindow* ui) {
    refresh(ui);
}

void RecommenderSystem::print_recommendation(Ui::MainWindow *ui) {
    cout << "Recommendation: " << endl;
    for (int u = 0; u < n_users; u++) {
        vector<float> recommended_items = recommend(u);
        if (recommended_items.size() > 0) {
            if (uuFC == 1) {
                QString myRecommendation = "Recommend item(s): ";
                cout << "    \t\tRecommend item(s):";
                for (int i = 0; i < recommended_items.size(); i++) {
                    cout << recommended_items[i] << " ";
                    myRecommendation = myRecommendation + QString("%1 ").arg(recommended_items[i]);
                }
                cout << "for user " << u << endl;
                myRecommendation = myRecommendation + QString("for user %1").arg(u);
                ui->userRecommendationListWidget->addItem(myRecommendation);
            }
            if (uuFC == 0) {
                cout << "    \t\tRecommend item " << u << " ";
                cout << "to user(s) : ";
                QString myRecommendation = QString("Recommend item %1 to user(s) : ").arg(u);
                for (int i = 0; i < recommended_items.size(); i++) {
                    cout << recommended_items[i] << " ";
                    myRecommendation = myRecommendation + QString("%1 ").arg(recommended_items[i]);
                }
                ui->itemRecommendationListWidget->addItem(myRecommendation);
                cout << endl;
            }
        }
    }
}

float RecommenderSystem::pred(int u, int i, int normalized = 1) {
    if (uuFC) return __pred(u, i, normalized);
    return __pred(i, u, normalized);
}

void RecommenderSystem::normalize_Y() {
    float users[N];
    assignArray(users, Y_data_T[0], N);
    assignMatrix(Ybar_data, Y_data);
    mu = createZerosVector(mu, n_users);

    for (int n = 0; n < n_users; n++) {
        // Ids
        vector< int > ids;
        for (int i = 0; i < N; i++)
            if (users[i] == n) ids.push_back(i);

        // item_ids
        vector< int > item_ids;
        for (int i = 0; i < ids.size(); i++) {
            item_ids.push_back(Y_data_T[1][ids[i]]);
        }

        // ratings
        vector< float > ratings;
        for (int i = 0; i < ids.size(); i++) {
            ratings.push_back(Y_data_T[2][ids[i]]);
        }

        // take mean ratings
        float m = average(ratings);
        mu[n] = m;

        // Ybar_data
        for (int i = 0; i < ids.size(); i++) {
            Ybar_data[ids[i]][2] = ratings[i] - mu[n];
        }
    }
    MatrixXd YbarTest = MatrixXd::Zero(n_items, n_users);
    for (int i = 0; i < N; i++) {
        YbarTest(int(Y_data[i][1]), int(Y_data[i][0])) = Ybar_data[i][2];
    }
    Ybar = YbarTest;
}

void RecommenderSystem::similarity(Ui::MainWindow* ui) {
    MatrixXd S_Ref(n_users, n_users);
    MatrixXd Ybar_T = Ybar.transpose();

    for (int i = 0; i < n_users; i++) {
        QString myPercent = QString("-------> Training : %1 %").arg(static_cast<float>(i) * 100 / static_cast<float>(n_users));
        cout << "-------> Training : " << static_cast<float>(i) * 100 / static_cast<float>(n_users) << "%" << endl;
        if (uuFC == 1) ui->userTraininglistWidget->addItem(myPercent);
        if (uuFC == 0) ui->itemTraininglistWidget->addItem(myPercent);

        for (int j = 0; j < n_users; j++) {
            vector<float> vec1;
            vector<float> vec2;
            for (int k = 0; k < n_items; k++) {
                vec1.push_back(Ybar_T(i, k));
                vec2.push_back(Ybar_T(j, k));
            }
            S_Ref(i, j) = cosine_similarity(vec1, vec2);
        }

    }
    cout << "-------> Training : " << 100 << "%" << endl;
    QString myPercent = QString("-------> Training : %1 %").arg(100);
    if (uuFC == 1) ui->userTraininglistWidget->addItem(myPercent);
    if (uuFC == 0) ui->itemTraininglistWidget->addItem(myPercent);
    S = S_Ref;
}

void RecommenderSystem::refresh(Ui::MainWindow* ui) {
//    ui->userTraininglistWidget->addItem("refresh ...........");
    normalize_Y();
    similarity(ui);
}

float RecommenderSystem::__pred(int u, int i, int normalized) {
    // Ids
    vector<float> ids;
    for (int k = 0; k < N; k++)
        if (int(Y_data[k][1]) == int(i)) ids.push_back(k);

    //users_rated_i
    vector<float > users_rated_i;
    for (int k = 0; k < ids.size(); k++) {
        users_rated_i.push_back(Y_data[int(ids[k])][0]);
    }


    // sim
    vector<float> sim;
    for (int k = 0; k < users_rated_i.size(); k++) {
        sim.push_back(S(u, int(users_rated_i[k])));
    }

    // a
    vector<float> a = slicing(argsort(sim), K);

    // nearest_s
    vector<float> nearest_s;
    for (int k = 0; k < a.size(); k++) {
        nearest_s.push_back(sim[int(a[k])]);
    }

    //////////////
    vector<float> r;
    for (int k = 0; k < a.size(); k++) {
        int index = users_rated_i[int(a[k])];
        r.push_back(Ybar(i, index));
    }

    // Return result
    float result;
    if (normalized == 1) {
        result = multiplyVectors(r, nearest_s) / (addVector(nearest_s) + 1e-8);
    }
    else {
        result = multiplyVectors(r, nearest_s) / (addVector(nearest_s) + 1e-8) + mu[u];
    }
    return result;
}

vector<float> RecommenderSystem::recommend(int u) {
    vector<float> recommended_items;

    // Ids
    vector< float > ids;
    for (int i = 0; i < N; i++)
        if (int(Y_data[i][0]) == int(u)) ids.push_back(i);

    //Items_rated_by_u
    vector< float > items_rated_by_u;
    for (int i = 0; i < ids.size(); i++) {
        items_rated_by_u.push_back(Y_data[int(ids[i])][1]);
    }

    //find predict rating and recommended
    for (int i = 0; i < n_items; i++) {
        if (!isInVector(items_rated_by_u, i)) {
            float rating = __pred(u, i);
            if (rating > ratingThreshold) recommended_items.push_back(i);
        }
    }
    return recommended_items;
}

float RecommenderSystem::addVector(vector<float> vec) {
    float result = 0;
    for (int i = 0; i < vec.size(); i++) result += abs(vec[i]);
    return result;
}

float RecommenderSystem::multiplyVectors(vector<float> vec1, vector<float> vec2) {
    float result = 0;
    for (int i = 0; i < vec1.size(); i++) result += vec1[i] * vec2[i];
    return result;
}

float RecommenderSystem::cosine_similarity(vector<float> vec1, vector<float> vec2) {
    float mul = 0.0, d_a = 0.0, d_b = 0.0;
    for (unsigned int i = 0; i < vec1.size(); ++i)
    {
        mul += vec1[i] * vec2[i];
        d_a += vec1[i] * vec1[i];
        d_b += vec2[i] * vec2[i];
    }
    if ((mul / (sqrt(d_a) * sqrt(d_b))) != (mul / (sqrt(d_a) * sqrt(d_b)))) {
        return 0.0;
    }
    return mul / (sqrt(d_a) * sqrt(d_b));
}

float RecommenderSystem::average(vector<float> vec) {
    if (vec.size() == 0) return 0;
    else {
        int sum = 0;
        for (int i = 0; i < vec.size(); i++) {
            sum += vec[i];
        }
        float mean = float(sum) / vec.size();
        return mean;
    }
}

void RecommenderSystem::displayVector(vector<float> vec) {
    cout << "==== Display Vector =====" << endl;
    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i] << " ";
    }
    cout << endl;
}
void RecommenderSystem::assignMatrix(float A[][M], float B[][M]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i][j] = B[i][j];
        }
    }
}
void RecommenderSystem::assignArray(float A[], float B[], int n) {
    for (int i = 0; i < n; i++) {
        A[i] = B[i];
    }
}
void RecommenderSystem::displayArray(int A[], int n) {
    cout << "==== Display Array =====" << endl;
    for (int i = 0; i < n; i++) {
        cout << A[i] << " ";
    }
    cout << endl;
}
void RecommenderSystem::displayMatrix(float matrix[][M]) {
    cout << "==== Display matrix ====" << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
void RecommenderSystem::displayMatrixT(float matrix[][N]) {
    cout << "==== Display matrix Trans ====" << endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
int RecommenderSystem::largest(float arr[], int n) {
    int i;
    int max = arr[0];
    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
    return max;
}
bool RecommenderSystem::isInVector(vector<float> vec1, int index) {
    for (int i = 0; i < vec1.size(); i++) {
        if (int(vec1[i]) == index) return true;
    }
    return false;
}
vector<float> RecommenderSystem::slicing(vector<float> vec, int K) {
    if (K > vec.size()) {
        return vec;
    }
    else {
        vector<float> result;
        for (int i = vec.size() - K; i < vec.size(); i++) result.push_back(vec[i]);
        return result;
    }
}
vector<float> RecommenderSystem::argsort(vector<float> array) {
    vector<float> indices(array.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
        [&array](int left, int right) -> bool {
        return array[left] < array[right];
    });
    return indices;
}
vector<float> RecommenderSystem::createZerosVector(vector<float> vec, int n) {
    for (int i = 0; i < n; i++) {
        vec.push_back(0);
    }
    return vec;
}
void RecommenderSystem::displayInfo() {
    cout << "=========== Recommend System =============" << endl;
}
void RecommenderSystem::displayNormalizedMatrix() {
    cout << "Nothing to display" << endl;
}
void RecommenderSystem::displaySimilarityMatrix() {
    cout << "Nothing to display" << endl;
}

// UserUserRecommendSystem
void UserUserRecommendSystem::displayInfo() {
    cout << "=========== User User Recommend System =============" << endl;
    cout << "Number of User : " << n_users << endl;
    cout << "Number of Item : " << n_items << endl;
}
void UserUserRecommendSystem::displayNormalizedMatrix() {
    cout << "=========== Normalized Matrix =============" << endl;
    cout << Ybar << endl;
}
void UserUserRecommendSystem::displaySimilarityMatrix() {
    cout << "=========== Similarity Matrix =============" << endl;
    cout << S << endl;
}

// ItemItemRecommendSystem
void ItemItemRecommendSystem::displayInfo() {
    cout << "=========== User User Recommend System =============" << endl;
    cout << "Number of User : " << n_users << endl;
    cout << "Number of Item : " << n_items << endl;
}
void ItemItemRecommendSystem::displayNormalizedMatrix() {
    cout << "=========== Normalized Matrix =============" << endl;
    cout << Ybar << endl;
}
void ItemItemRecommendSystem::displaySimilarityMatrix() {
    cout << "=========== Similarity Matrix =============" << endl;
    cout << S << endl;
}










