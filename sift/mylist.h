#ifndef SIFT_MYLIST_H
#define SIFT_MYLIST_H

#include <vector>

template <typename T>
class MyList {
public:
    std::vector<T> data;

    MyList() {
        data.clear();
    }

    ~MyList() {
        data.clear();
    }

    void push_back(T value) {
        data.push_back(value);
    }
};


#endif //SIFT_MYLIST_H
