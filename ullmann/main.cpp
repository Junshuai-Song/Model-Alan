//
//  main.cpp
//  ullmann
//
//  Created by songjs on 16/10/29.
//  Copyright © 2016年 songjs. All rights reserved.
//

#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <set>
#include <string.h>
#include <string>
#include <fstream>

using namespace std;

typedef struct Edge{
    int from,to,elabel;
}Edge;

typedef struct Vertex{
    int id,label;
    vector<Edge> edge;
    void push(int from,int to,int elabel){
        edge.resize(edge.size() + 1);
        edge[edge.size() - 1].from = from;
        edge[edge.size() - 1].to = to;
        edge[edge.size() - 1].elabel = elabel;
        return ;
    }

}Vector;

class Graph: public vector<Vertex>{
private:
    unsigned int edge_size_;
    
public:
    int id;
    typedef vector<Vertex>::iterator vertex_iterater;
    unsigned int edge_size(){ return edge_size_; }
    unsigned int vertex_size(){ return (unsigned int)size();}
    int **M;
    
    void build(){
        // 邻接矩阵
//        cout<<"vertex_size="<<vertex_size()<<endl;
        M = new int*[vertex_size()];
        for(int i=0;i<vertex_size();i++) M[i] = new int[vertex_size()];
        for(int i=0;i<vertex_size();i++) for(int j=0;j<vertex_size();j++) M[i][j] = 0;
        
        for(vertex_iterater iter = (*this).begin(); iter!=(*this).end(); iter++){
            for(vector<Edge>::iterator iter2 = (*iter).edge.begin(); iter2!=(*iter).edge.end(); iter2++){
                M[(*iter2).from][(*iter2).to] = 1; //无向图
                M[(*iter2).to][(*iter2).from] = 1;
            }
        }
    }
    int find_label(int n){
        // 找到顶点n的label
        return (*this)[n].label;
    }
    int count(int n){
        return (int)(*this)[n].edge.size();
    }
    void print(){
        for(vertex_iterater iter = (*this).begin(); iter!=(*this).end(); iter++){
            for(vector<Edge>::iterator iter2 = (*iter).edge.begin(); iter2!=(*iter).edge.end(); iter2++){
                cout<<(*iter2).from<<" "<<(*iter2).to<<endl;
            }
        }
    }
};

class Ullmann{
public:
    vector<Graph> graph;
    vector<Graph> query_graph;
    
public:
    char *filename1,*filename2;
    int answer_number;
    clock_t start,end;
    int **MA,**MB,**M,**M_,**MC;
    int *vis;
    
    void read(char filename[], vector<Graph> &graph_){
        ifstream in; in.open(filename);
        string line; Graph g; g.clear(); int id=0;
        while(in>>line){
            if(line[0]=='t'){
                getline(in, line);
                if(!g.empty()){
                    // 新的一个图，加进来
                    g.build();  //构建邻接矩阵
                    g.id = id++;
//                    cout<<g.vertex_size()<<endl;
                    graph_.push_back(g);
                    g.clear();
                }
                if(line[4]=='-') break; // end of the file
            }
            else if(line[0]=='v'){
                int id_,label;
                in>>id_>>label;
                getline(in,line);
                g.resize(max((int)g.size(), id_+1));
                g[id_].label = label;
            }else if(line[0]=='e'){
                int from,to,elabel;
                in>>from>>to>>elabel;
                g.resize(max((int)g.size(), from+1));
                g.resize(max((int)g.size(), to+1));
                g[from].push(from, to, elabel);
                g[to].push(to, from, elabel);
                getline(in, line);
            }
        }
        in.close();
    }
    
    void pre_check(Graph &Q, Graph &G){
        MA = Q.M; MB = G.M;
        // 构建M和M_
        M = new int*[Q.vertex_size()];
        M_ = new int*[Q.vertex_size()];
        MC = new int*[Q.vertex_size()];
        for(int i=0;i<Q.vertex_size();i++) {
            M[i] = new int[G.vertex_size()];
            M_[i] = new int[G.vertex_size()];
            MC[i] = new int[Q.vertex_size()];
        }
        
        // 之后使用各种规则尽可能减少M中1的个数
        for(int i=0;i<Q.vertex_size();i++){
            for(int j=0;j<G.vertex_size();j++){
                M[i][j] = 1;
                if(Q.find_label(i) != G.find_label(j)){ M[i][j] = 0;continue;}
                if(Q.count(i) > G.count(j)){ M[i][j] = 0;continue; }
            }
        }
        int num = 1;
        while(num>0){   //这个过程迭代进行
            num = 0;
            for(int i=0;i<Q.vertex_size();i++){
                for(int j=0;j<G.vertex_size();j++){
                    if(M[i][j]==1){
                        int flag=1;
                        for(int x=0;x<Q.vertex_size();x++){
                            // 与i有关联的任意一条边，都要存在
                            if(MA[i][x]==0) continue;
                            int t = 0;
                            for(int y=0;y<G.vertex_size();y++){
//                                if(M[x][y]*MB[y][j]==1 && Q.find_label(i)==G.find_label(j)){
                                if(M[x][y]*MB[y][j]==1){
                                    t=1; break;
                                }
                            }
                            if(t==1) continue;
                            else {flag=0;break;}
                        }
                        M[i][j]=flag;   //更新M矩阵
                        if(flag==0) num++;
                    }
                }
            }
        }
    }
    void after_check(Graph &Q){
        // 销毁矩阵M
        for(int i=0;i<Q.vertex_size();i++){
            delete[] M[i]; delete[] M_[i]; delete[] MC[i];
        }
        delete[] M;
        delete[] M_;
        delete[] MC;
    }
    int **deal(Graph &Q,Graph &G){
        for(int i=0;i<Q.vertex_size();i++) for(int j=0;j<Q.vertex_size();j++) MC[i][j] = 0;
//        memset(MC,0,sizeof(MC));
        // 进行矩阵乘法：M_ * 转置(M_ * MB)
        int **M__ = new int*[Q.vertex_size()];
        for(int i=0;i<Q.vertex_size();i++) M__[i] = new int[G.vertex_size()];
        for(int i=0;i<Q.vertex_size();i++){
            for(int j=0;j<G.vertex_size();j++){
                int tot = 0;
                for(int k=0;k<G.vertex_size();k++){
                    tot += M_[i][k]*MB[k][j];
                    if(tot>0) break;
                }
                M__[i][j] = tot;
            }
        }
        for(int i=0;i<Q.vertex_size();i++){
            for(int j=0;j<Q.vertex_size();j++){
                int tot=0;
                for(int k=0;k<G.vertex_size();k++){
                    tot += M_[i][k]*M__[j][k];
                    if(tot>0) break;
                }
                MC[i][j] = tot;
            }
        }
        for(int i=0;i<Q.vertex_size();i++){
            delete[] M__[i];
        }delete[] M__;
        return MC;
    }
    bool dfs(Graph &Q, Graph &G, int t){ //t表示找到图Q中的第几个顶点
        // 只需要用到矩阵MA、MB和M，递归M找所有候选集，之后利用MA、MB进行判断
        if(t==Q.vertex_size()){
            // 最后一行选择完了，边界判断当前M_矩阵是否为一个好的映射
            int **MC = deal(Q,G); // = M_ * 转职(M_ * MB)
            for(int i=0;i<Q.vertex_size();i++){
                for(int j=0;j<Q.vertex_size();j++){
                    if(MA[i][j]==1){
                        if(MC[i][j]==1) continue;
                        else {
//                            cout<<"MC不满足要求"<<endl;
                            return false;
                        }
                    }
                }
            }
            
//            cout<<"---"<<endl;
            // 输出当前映射结果集M_
            // 注意：需要边label也满足要求
            int *v = new int[Q.vertex_size()];
            for(int i=0;i<Q.vertex_size();i++){
                for(int j=0;j<G.vertex_size();j++){
                    if(M_[i][j]==1){
                        v[i] = j;
                    }
                }
            }
            int flag=1;
            for(int i=0;i<Q.vertex_size();i++){
                for(int j=0;j<Q.vertex_size();j++){
                    if(MA[i][j]==1){
                        int lable = -1;
                        for(vector<Edge>::iterator iter = Q[i].edge.begin(); iter!=Q[i].edge.end(); iter++){
                            if((*iter).to==j){
                                lable = (*iter).elabel;
                                break;
                            }
                        }
                        int lable_ = -1;
                        for(vector<Edge>::iterator iter = G[v[i]].edge.begin(); iter!=G[v[i]].edge.end(); iter++){
                            if((*iter).to==v[j]){
                                lable_ = (*iter).elabel;
                                break;
                            }
                        }
                        if(lable == lable_) continue;
                        else{
                            flag=0;
                            i = Q.vertex_size();
                            break;
                        }
                    }
                }
            }
            if(flag==0) return false;   //当前映射，边lable不对应，不输出
            
            cout<<"查询图id #"<<Q.id<<" 映射数据库中图id #"<<G.id<<"  number: "<<answer_number++<<endl;
            for(int i=0;i<Q.vertex_size();i++){
                for(int j=0;j<G.vertex_size();j++){
                    if(M_[i][j]==1){
                        cout<<i<<" -> "<<j<<endl; break;
                    }
                }
            }
            cout<<endl;
            return true;
        }else{
            // 寻找结果集
            for(int i=0;i<G.vertex_size();i++){
                if(M[t][i]==1 && vis[i]==0){
                    M_[t][i] = 1;
                    vis[i] = 1;
                    dfs(Q,G,t+1);
                    M_[t][i] = 0;
                    vis[i] = 0;
                }
            }
            return true;
        }
    }
    void check(Graph &Q, Graph &G){
        // 子图同构匹配，如果匹配则直接输出，answer_number++；否则直接return
        if(Q.edge_size() > G.edge_size()) return ;
        pre_check(Q,G);    //使用顶点 & refinement规则尽可能减少M矩阵中的
        
        int flag=1;
        for(int i=0;i<Q.vertex_size();i++){
            int num=0;
            for(int j=0;j<G.vertex_size();j++){
                num += M[i][j];
            }
            if(num==0) {flag=0; break;}
        }
        if(flag == 0) return;   //提前剪枝，存在某一行没有1，无解
        
        vis = new int[G.vertex_size()];
        for(int i=0;i<G.vertex_size();i++) vis[i]=0;
        for(int i=0;i<Q.vertex_size();i++) for(int j=0;j<G.vertex_size();j++) M_[i][j] = 0;
        for(int i=0;i<Q.vertex_size();i++){
            for(int j=0;j<G.vertex_size();j++){
                cout<<" "<<M[i][j];
            }cout<<endl;
        }
        dfs(Q,G,0);
        after_check(Q);
    }
    void run(){
        // 每次拿query_graph图中的一个子图，来与数据库中某一个子图进行对比
        int num=0;
        for(vector<Graph>::iterator iter = query_graph.begin(); iter!=query_graph.end(); iter++){
            Graph Q = (*iter);
            int number = 0;
            for(vector<Graph>::iterator iter2 = graph.begin(); iter2!=graph.end(); iter2++){
                Graph G = (*iter2);
                // 保证Q是小图，进行同构匹配
                if(Q.vertex_size() > G.vertex_size()) continue;
                check(Q,G);
                number++;
                if(number%1==0)
                    cout<<"G_number="<<number<<endl;
            }
            num++;
            if(num % 1==0)
                cout<<"Q_num="<<num<<endl;
        }
    }
    
    Ullmann(){
        start = clock();
        filename1 = new char[150];
        filename2 = new char[150];
        cout<<"请输入图数据库文件路径:(150字符以内)"<<endl;
//        cin>>filename2;
        strcpy(filename1, "/Users/songjs/Desktop/workspace/ullmann/ullmann/mygraphdb.data");
        strcpy(filename2, "/Users/songjs/Desktop/workspace/ullmann/ullmann/Q4.my");
//        strcpy(filename1, "/home/songjunshuai/workspace/ullmann/mygraphdb.data");
//        strcpy(filename2, "/home/songjunshuai/workspace/ullmann/Q8.my");
        read(filename1, graph);
        read(filename2, query_graph);
        cout<<graph.size()<<" "<<query_graph.size()<<endl;
        answer_number = 0;
    }
    
    ~Ullmann(){
        delete[] filename1;
        delete[] filename2;
    }
    void run_time(){
        end = clock();
        cout<<"run time : "<<(double)(end-start)/CLOCKS_PER_SEC<<"s."<<endl;
    }
};

int main() {
    Ullmann ullmann;
    ullmann.run();
    ullmann.run_time();
    
    return 0;
}
























