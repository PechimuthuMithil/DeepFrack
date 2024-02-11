#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <json/json.h>
#include <chrono>

using namespace std;

void generateCombinations(vector<int>& currentCombination, int remainingArea, int remainingWidth, int smallestSquare, vector<vector<int>>& combinations) {
    if (remainingArea == 0) {
        combinations.push_back(currentCombination);
        return;
    }
    else if (remainingArea < 0 || remainingWidth == 0) {
        return;
    }

    for (int i = smallestSquare; i <= min(remainingWidth, static_cast<int>(sqrt(remainingArea))); i++) {
        currentCombination.push_back(i);
        generateCombinations(currentCombination, remainingArea - i * i, min(i, remainingArea - i * i), i, combinations);
        currentCombination.pop_back();
    }
}

void getSquareCombinations(int width, int smallestSquare, vector<vector<int>>& combinations) {
    vector<int> currentCombination;
    generateCombinations(currentCombination, width * width, width, smallestSquare, combinations);
}

int main() {
    int smallest_square = 2; // Specify the smallest square to be considered
    int max_width = 550;
    auto start_time = std::chrono::high_resolution_clock::now();
    Json::Value results;

    for (int width = 1; width <= max_width; width++) {
        vector<vector<int>> combinations;
        getSquareCombinations(width, smallest_square, combinations);

        Json::Value widthCombinations(Json::arrayValue);
        for (const auto& combination : combinations) {
            Json::Value jsonCombination(Json::arrayValue);
            for (const auto& square : combination) {
                jsonCombination.append(square);
            }
            widthCombinations.append(jsonCombination);
        }

        results[to_string(width)] = widthCombinations;
    }

    // Save the results to a JSON file
    ofstream outputFile("/workspace/CheatSheet_test1.json");
    if (outputFile.is_open()) {
        outputFile << results;
        outputFile.close();
        cout << "Results saved to output.json" << endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        // Calculate the elapsed time duration
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        // Print the elapsed time in seconds
        std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
    }
    else {
        cout << "Failed to open output.json for writing" << endl;
    }

    return 0;
}
