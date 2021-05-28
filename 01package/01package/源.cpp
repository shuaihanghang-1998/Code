#include <stdio.h>
#include <queue>
#include <vector>
#define _CRT_SECURE_NO_WARNINGS
using namespace std;
class Object
{
public:
    int id;
    int weight;
    int price;
    float d;
};
class MaxHeapQNode
{
public:
    MaxHeapQNode* parent;
    int lchild;
    int upprofit;
    int profit;
    int weight;
    int lev;
};
struct cmp
{
    bool operator()(MaxHeapQNode*& a, MaxHeapQNode*& b) const
    {
        return a->upprofit < b->upprofit;
    }
};
bool compare(const Object& a, const Object& b)
{
    return a.d >= b.d;
}
int n;
int c;
int cw;
int cp;
int bestp;
Object obj[100];
int bestx[100];
void InPut()
{
    scanf("%d %d", &n, &c);
    for (int i = 1; i <= n; ++i)
    {
        scanf("%d %d", &obj[i].price, &obj[i].weight);
        obj[i].id = i;
        obj[i].d = 1.0 * obj[i].price / obj[i].weight;
    }

    sort(obj + 1, obj + n + 1, compare);
    //    for(int i = 1; i <= n; ++i)
    //        cout << obj[i].d << " ";
    //    cout << endl << "InPut Complete" << endl;
}
int Bound(int i)
{
    int tmp_cleft = c - cw;
    int tmp_cp = cp;
    while (tmp_cleft >= obj[i].weight && i <= n)
    {
        tmp_cleft -= obj[i].weight;
        tmp_cp += obj[i].price;
        i++;
    }
    if (i <= n)
    {
        tmp_cp += tmp_cleft * obj[i].d;
    }
    return tmp_cp;
}
void AddAliveNode(priority_queue<MaxHeapQNode*, vector<MaxHeapQNode*>, cmp>& q, MaxHeapQNode* E, int up, int wt, int curp, int i, int ch)
{
    MaxHeapQNode* p = new MaxHeapQNode;
    p->parent = E;
    p->lchild = ch;
    p->weight = wt;
    p->upprofit = up;
    p->profit = curp;
    p->lev = i + 1;
    q.push(p);
    //    cout << "��������ϢΪ " << endl;
    //    cout << "p->lev = " << p->lev << " p->upprofit = " << p->upprofit << " p->weight =  " << p->weight << " p->profit =  " << p->profit << endl;
}
void MaxKnapsack()
{
    priority_queue<MaxHeapQNode*, vector<MaxHeapQNode*>, cmp > q; // �󶥶�
    MaxHeapQNode* E = NULL;
    cw = cp = bestp = 0;
    int i = 1;
    int up = Bound(1); //Bound(i)�����������i��δ����ʱ�������ֵ
    while (i != n + 1)
    {
        int wt = cw + obj[i].weight;
        if (wt <= c)
        {
            if (bestp < cp + obj[i].price)
                bestp = cp + obj[i].price;
            AddAliveNode(q, E, up, cw + obj[i].weight, cp + obj[i].price, i, 1);
        }
        up = Bound(i + 1); //ע������ up != up - obj[i].price���� up >= up - obj[i].price
        if (up >= bestp) //ע����������Ǵ��ڵ���
        {
            AddAliveNode(q, E, up, cw, cp, i, 0);
        }
        E = q.top();
        q.pop();
        cw = E->weight;
        cp = E->profit;
        up = E->upprofit;
        i = E->lev;
    }
    for (int j = n; j > 0; --j)
    {
        bestx[obj[E->lev - 1].id] = E->lchild;
        E = E->parent;
    }
}
void OutPut()
{
    printf("����װ����Ϊ: %d\n", bestp);
    printf("װ�����ƷΪ: \n");
    printf("(   ");
    for (int i = 1; i <= n; ++i)
        if (bestx[i] == 1)
            printf("%d    ", i);
    printf("   )");
}
int main()
{
    InPut();
    MaxKnapsack();
    OutPut();
}
