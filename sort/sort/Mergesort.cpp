#include<iostream>
#include<stdio.h>
#include<time.h>
#define random(x) (rand()%x)
#include <ctime>

using namespace std;
template<class T>void QuickSort(T a[], int p, int r);
template<class T>int Partition(T a[], int p, int r);
template<class T>void MergeSort(T a[], int left, int right);
template<class T>void MergeSort(T a[], T b[], int left, int right);
template<class T>void Merge(T c[], T d[], int l, int m, int r);
template<class T>void Copy(T a[], T b[], int l, int r);
int main()
{
	clock_t startTime, endTime;
	srand((unsigned)time(NULL));
	int const n(100000);
	int a[n] = {0};
	int b[n] = { 0 };
	cout << "Input " << n << "numbers " << endl;
	for (int i = 0; i < n; i++)
		a[i] = rand() % (5 * n);
	//for(int j=0;j<n;j++) 
	//b[j]=a[j]; 
	//cout << "The initial array is:" << endl;
	//for (int i = 0; i < n; i++)
	//	cout << a[i] << "  ";
	//cout << endl;
	cout << "---------------QuickSort---------------" << endl;
	startTime = clock();
	QuickSort(a, 0, n - 1);
	endTime = clock();
	cout << "run time(ms): " << endTime - startTime << endl;
	//cout << "The sorted array is:" << endl;
	//for (int i = 0; i < n; i++)
	//	cout << a[i] << "  ";
	//cout << endl;
	cout << "---------------MergeSort---------------" << endl;
	startTime = clock();
	MergeSort(a, b ,0, n - 1);
	endTime = clock();
	cout << "run time(ms): " << endTime - startTime << endl;
	//cout << "The sorted array is:" << endl;
	//for (int i = 0; i < n; i++)
	//	cout << a[i] << "  ";
	//cout << endl;
	//return 0;
}
template<class T>
void MergeSort(T a[], T b[] ,int left, int right) // 
{
	if (left < right)
	{
		int i = (left + right) / 2;
		MergeSort(a, b , left, i);
		MergeSort(a, b , i + 1, right);
		Merge(a, b, left, i, right);
		Copy(a, b, left, right);
	}
}
template<class T>
void MergeSort(T a[], int left, int right) // 
{
	if (left < right)
	{
		int i = (left + right) / 2;
		T* b = new T[right];
		MergeSort(a, left, i);
		MergeSort(a, i + 1, right);
		Merge(a, b, left, i, right);
		Copy(a, b, left, right);
	}
}
template<class T>
void Merge(T c[], T d[], int l, int m, int r)
{
	int i = l;
	int j = m + 1;
	int k = l;
	while ((i <= m) && (j <= r))
	{
		if (c[i] <= c[j])d[k++] = c[i++];
		else d[k++] = c[j++];
	}
	if (i > m)
	{
		for (int q = j; q <= r; q++)
			d[k++] = c[q];
	}
	else
		for (int q = i; q <= m; q++)
			d[k++] = c[q];
}
template<class T>
void Copy(T a[], T b[], int l, int r)
{
	for (int i = l; i <= r; i++)
		a[i] = b[i];
}
template<class T>
inline void Swap(T& s, T& t)
{
	T temp = s;
	s = t;
	t = temp;
}
template<class T>
int Partition(T a[], int p, int r)
{
	int i = p, j = r + 1;
	T x = a[p];
	while (true)
	{
		while (a[++i] < x);
		while (a[--j] > x);
		if (i >= j)break;
		Swap(a[i], a[j]);
	}
	a[p] = a[j];
	a[j] = x;
	return j;
}
template<class T>
void QuickSort(T a[], int p, int r)
{
	if (p < r)
	{
		int q = Partition(a, p, r);
		QuickSort(a, p, q - 1);
		QuickSort(a, q + 1, r);
	}
}
