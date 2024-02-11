#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void generateCombinations(vector<int>& currentCombination, int remainingArea, int remainingWidth, int smallestSquare) {
    if (remainingArea == 0) {
        for (const auto& square : currentCombination) {
            cout << square << " ";
        }
        cout << endl;
        return;
    }
    else if (remainingArea < 0 || remainingWidth == 0) {
        return;
    }

    for (int i = smallestSquare; i <= min(remainingWidth, static_cast<int>(sqrt(remainingArea))); i++) {
        currentCombination.push_back(i);
        generateCombinations(currentCombination, remainingArea - i * i, min(i, remainingArea - i * i), i);
        currentCombination.pop_back();
    }
}

void getSquareCombinations(int a, int b, int smallestSquare) {
    vector<int> currentCombination;

    generateCombinations(currentCombination, a * b, max(a, b), smallestSquare);
}

int main() {
    int rectangle_a = 10;
    int rectangle_b = 10;
    int smallest_square = 2; // Specify the smallest square to be considered
    for (int i = 1; i < 201; i++){
        getSquareCombinations(i, i, smallest_square);
    }

    return 0;
}


