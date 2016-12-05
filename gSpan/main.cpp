//
//  main.cpp
//  gSpan
//
//  Created by songjs on 16/10/17.
//  Copyright © 2016年 songjs. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <string.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <stack>
#include <fstream>
#include <assert.h>
#include <ctime>

using namespace std;

template<class T> inline void _swap(T &x, T &y){
    T z = x;
    x = y;
    y = z;
}
struct Edge{
    int from,to,elabel; // [from, to , elabel]
    unsigned int id;    // edge id
    Edge(): from(0), to(0), elabel(0), id(0){};
};

class Vertex{
public:
    typedef vector<Edge>::iterator edge_iterater;
    int label;
    vector<Edge> edge;  //  ?
    void push(int from, int to, int elabel){
        edge.resize(edge.size() + 1);
        edge[edge.size()-1].from = from;
        edge[edge.size()-1].to = to;
        edge[edge.size()-1].elabel = elabel;
        return ;
    }
    
};



class Graph: public vector<Vertex>{
private:
    unsigned int edge_size_;
public:
    typedef vector<Vertex>::iterator vertex_iterater;
    Graph(){}
    unsigned int edge_size(){ return edge_size_; }
    unsigned int vertex_size(){ return (unsigned int)size();}
    void buildEdge(){
        char buf[512];
        map<string, int> tmp;       // tmp只是临时用一下
        unsigned int id = 0;
        for(int from=0; from<(int)size(); ++from){
            for(Vertex::edge_iterater it = (*this)[from].edge.begin(); it!=(*this)[from].edge.end(); it++){
                if(from <= it->to) sprintf(buf, "%d %d %d", from, it->to, it->elabel);
                else sprintf(buf, "%d %d %d", it->to, from, it->elabel);
                
                if(tmp.find(buf) == tmp.end()){
                    it->id = id;
                    tmp[buf] = id++;
                }else{
                    it->id = tmp[buf];
                }
            }
        }
        edge_size_ = id;
    }
    
};

struct PDFS{
    unsigned int id;    //原始输入图id
    Edge *edge;
    PDFS *prev;
    PDFS(): id(0), edge(0), prev(0){};
};

class Projected: public vector<PDFS>{
public:
    void push(int id, Edge *edge, PDFS *prev){
        resize(size() + 1);
        PDFS &d = (*this)[size()-1];
        d.id = id; d.edge = edge; d.prev = prev;
    }
};



typedef vector<Edge *> EdgeList;

bool get_forward_root(Graph &g, Vertex &v, EdgeList &result){
    result.clear();
    for(Vertex::edge_iterater it = v.edge.begin(); it!=v.edge.end(); it++){
        assert(it->to >=0 && it->to <g.size());
        if(v.label <= g[it->to].label)
            result.push_back(&(*it));
    }
    return (!result.empty());
}


class DFS{
public:
    int from,to,fromlabel,elabel,tolabel;
    friend bool operator == (const DFS &a, const DFS &b){
        return (a.from==b.from && a.to==b.to && a.fromlabel==b.fromlabel && a.tolabel==b.tolabel && a.elabel==b.elabel);
    }
    friend bool operator != (const DFS &a, const DFS &b){ return (!(a==b));}
    DFS(): from(0), to(0), fromlabel(0), elabel(0), tolabel(0){};
};

typedef vector<int> RMPath; //最右路径

struct DFSCode: public vector<DFS>{
private:
    RMPath rmpath;
public:
    const RMPath& buildRMPath(){
        rmpath.clear();
        int old_from = -1;
        
        for(int i=(int)size()-1 ;i>=0; i--){
            // 前向边
            if((*this)[i].from < (*this)[i].to && (rmpath.empty() || old_from == (*this)[i].to)){
                rmpath.push_back(i);
                old_from = (*this)[i].from;
            }
        }
        return rmpath;
    }
    
    // 将当前DFS编码转为图
    bool toGraph(Graph &g){
        g.clear();
        for(DFSCode::iterator it = begin(); it!=end(); it++){
            g.resize(max(max(it->from, it->to), (int)g.size()) + 1);    // a bug before.
            
            if(it->fromlabel != -1)
                g[it->from].label = it->fromlabel;
            if(it->tolabel != -1)
                g[it->to].label = it->tolabel;
            g[it->from].push(it->from, it->to, it->elabel);
            g[it->to].push(it->to, it->from, it->elabel);
        }
        g.buildEdge();
        return true;
    }
    
    // 从当前图重建DFS编码
    bool fromGraph(Graph &g);
    
    // 返回图中节点数目
    unsigned int nodeCound(void){
        unsigned int nodecount = 0;
        for(DFSCode::iterator it = begin(); it!=end(); it++)
            nodecount = max(nodecount, (unsigned int)(max(it->from, it->to)+1));
        return nodecount;
    }
    
    void push(int from, int to, int fromlabel, int elabel, int tolabel){
        resize(size() + 1);
        DFS &d = (*this)[size() - 1];
        d.from = from; d.to = to; d.fromlabel=fromlabel; d.elabel = elabel; d.tolabel = tolabel;
    }
    void pop(){
        resize(size() - 1);
    }
    
    
};

class History: public vector<Edge*>{
private:
    vector<int> edge;
    vector<int> vertex;
    
public:
    bool hasEdge(unsigned int id){ return (bool)edge[id];}
    bool hasVertex(unsigned int id){ return (bool)vertex[id];}
    void build(Graph &graph, PDFS *e){
        clear();edge.clear(); edge.resize(graph.edge_size());
        vertex.clear(); vertex.resize(graph.size());
        if(e){
            push_back(e->edge);
            edge[e->edge->id] = vertex[e->edge->from] = vertex[e->edge->to] = 1;        //1 表示当前顶点或者边访问过！那么下次再继续
            for(PDFS *p = e->prev; p ;p=p->prev){
                push_back(p->edge);
                edge[p->edge->id] = vertex[p->edge->from] = vertex[p->edge->to] = 1;
            }
            reverse(begin(), end());    //vector 翻转
        }
        
    }
    History(){}
    History(Graph &g, PDFS *p){ build(g,p);}
};

Edge *get_backward(Graph &graph, Edge* e1, Edge* e2, History &history){
    if(e1 == e2) return 0;
    
    for(Vertex::edge_iterater it = graph[e2->to].edge.begin(); it!=graph[e2->to].edge.end(); it++){
        if(history.hasEdge(it->id)) continue;
        if((it->to == e1->from) && ((e1->elabel < it->elabel) || ((e1->elabel == it->elabel) && (graph[e1->to].label <= graph[e2->to].label)))){
            return &(*it);
        }
    }
    return 0;
}

bool get_forward_pure(Graph &graph, Edge *e, int minlabel, History &history, EdgeList &result){
    result.clear();
    
    for(Vertex::edge_iterater it=graph[e->to].edge.begin(); it!=graph[e->to].edge.end(); it++){
        assert(it->to >=0 && it->to < graph.size());
        if(minlabel > graph[it->to].label || history.hasVertex(it->to))
            continue;
        result.push_back(&(*it));
    }
    return (!(result.empty()));
}

bool get_forward_rmpath(Graph &graph, Edge *e, int minlabel, History &history, EdgeList &result){
    result.clear();
    int tolabel = graph[e->to].label;
    
    for(Vertex::edge_iterater it=graph[e->from].edge.begin(); it!=graph[e->from].edge.end(); it++){
        int tolabel2 = graph[it->to].label;
        if(e->to == it->to || minlabel>tolabel2 || history.hasVertex(it->to))
            continue;
        if(e->elabel < it->elabel || (e->elabel == it->elabel && tolabel <= tolabel2))
            result.push_back(&(*it));
    }
    return (!result.empty());
}

class gSpan{
private:
    typedef map<int, map<int, map<int, Projected> > > Projected_map3;
    typedef map<int, map<int, Projected> >           Projected_map2;
    typedef map<int, Projected>                     Projected_map1;
    typedef map<int, map<int, map<int, Projected> > >::iterator Projected_iterator3;
    typedef map<int, map<int, Projected> >::iterator           Projected_iterator2;
    typedef map<int, Projected>::iterator                     Projected_iterator1;
    typedef map<int, map<int, map<int, Projected> > >::reverse_iterator Projected_riterator3;
    
    vector<Graph> TRANS;
    DFSCode DFS_CODE;
    DFSCode DFS_CODE_IS_MIN;
    Graph   GRAPH_IS_MIN;
    
public:
    char *filename; // filename
    int minsup;
    int maxpat_max,maxpat_min;  // the number of nodes in the fre.. graph.
    
    
    /*
     singular vertex.
     */
    map<unsigned int, map<unsigned int, unsigned int> > singleVertex;
    map<unsigned int, unsigned int> singleVertexLabel;
    
    int answer_number = 0;
    
    clock_t start,end;
public:
    
    gSpan(){
        filename = new char[150];
        
        
    }
    gSpan(int argc, const char * argv[]){
        filename = new char[150];
        cout<<"请输入图文件路径:(150字符以内)"<<endl;
        cin>>filename;
//        cout<<"请输入最小支持度、挖掘频繁子图中最少与最多节点个数(无限制请输入-1)："<<endl;
        cout<<"请输入最小支持度(无限制请输入-1)："<<endl;
        cin>>minsup;    //>>maxpat_min>>maxpat_max;
        if(maxpat_min == -1) maxpat_min = 0;
        if(maxpat_max == -1) maxpat_max = 0xffffffff;
        
//        strcpy(filename, "/Users/songjs/Desktop/workspace/gSpan/gSpan/graph.data");
        start = clock();
        read();
    }
    ~gSpan(){
        delete[] filename;
    }
    void read(){
        ifstream in; in.open(filename);
        
        vector<string> result;
        string line; Graph g; g.clear();
        while(in>>line){
            if(line[0]=='t'){
                getline(in, line);
                if(!g.empty()){
                    // 新的一个图，加进来
                    g.buildEdge();  // 给edge编号，赋值id
                    TRANS.push_back(g);
                    g.clear();
                }
                if(line[4]=='-') break; // end of the file
            }
            else if(line[0]=='v'){
                int id,label;
                in>>id>>label;
                getline(in,line);
                g.resize(max((int)g.size(), id+1));
                g[id].label = label;
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
        
    }
    unsigned int support(Projected &projected){
        // 计算当前子图的频繁数
        unsigned int oid = 0xffffffff;
        unsigned int size = 0;
        for(Projected::iterator cur = projected.begin(); cur!=projected.end(); cur++){
            if(oid != cur->id){
                size++;
            }
            oid = cur->id;
        }
        return size;
        
    }
    bool project_is_min(Projected &projected){
        // projected中有当前子图的多条边，从每一条开始尝试进行判断
        const RMPath & rmpath = DFS_CODE_IS_MIN.buildRMPath();      // rmpath是倒序，0的话就是min Code的搜索起始点
        int minlabel          = DFS_CODE_IS_MIN[0].fromlabel;       // 这里DFS_CODE_IS_MIN是有序的，在map中会自动按照label排序
        int maxtoc            = DFS_CODE_IS_MIN[rmpath[0]].to;
        {
            Projected_map1 root;
            bool flg = false;
            int newto = 0;
            
            for(int i=(int)rmpath.size()-1; !flg && i>=1; i--){     // [5,4,3,0]
                for(unsigned int n=0;n<projected.size(); n++){  // 这里存的是DFS_CODE_IS_MIN中第一条边 的所有同label[froml,el,tol]的边的位置
                    PDFS *cur = &projected[n];
                    History history(GRAPH_IS_MIN, cur);
                    Edge *e = get_backward(GRAPH_IS_MIN, history[rmpath[i]], history[rmpath[0]], history);
                    if(e){
                        root[e->elabel].push(0, e, cur);
                        newto = DFS_CODE_IS_MIN[rmpath[i]].from;
                        flg = true;
                    }
                }
            }
            if(flg){
                Projected_iterator1 elabel = root.begin();
                DFS_CODE_IS_MIN.push(maxtoc, newto, -1, elabel->first, -1);
                
                // 剪枝，新的DFS_CODE不断增多，每次都要和MIN_CODE相同才继续向下找。
                if(DFS_CODE[DFS_CODE_IS_MIN.size()-1] != DFS_CODE_IS_MIN[DFS_CODE_IS_MIN.size() - 1]) return false;
                return project_is_min(elabel->second);
            }
            
        }
        {
            // 找forward
            bool flg = false;
            int newfrom = 0;
            Projected_map2 root;
            EdgeList edges;
            for(unsigned int n=0; n<projected.size(); n++){
                PDFS *cur = &projected[n];
                History history(GRAPH_IS_MIN, cur);
                if(get_forward_pure(GRAPH_IS_MIN, history[rmpath[0]], minlabel, history, edges)){
                    flg = true;
                    newfrom = maxtoc;
                    for(EdgeList::iterator it = edges.begin(); it!=edges.end(); it++){
                        root[(*it)->elabel][GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
                    }
                }
            }
            for(int i=0; !flg && i<(int)rmpath.size(); i++){
                for(unsigned int n=0;n<projected.size(); n++){
                    PDFS *cur = &projected[n];
                    History history(GRAPH_IS_MIN,cur);
                    if(get_forward_rmpath(GRAPH_IS_MIN, history[rmpath[i]], minlabel, history, edges)){
                        flg = true;
                        newfrom = DFS_CODE_IS_MIN[rmpath[i]].from;
                        for(EdgeList::iterator it = edges.begin(); it!=edges.end(); it++){
                            root[(*it)->elabel][GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
                        }
                    }
                }
            }
            if(flg){
                Projected_iterator2 elabel = root.begin();
                Projected_iterator1 tolabel = elabel->second.begin();
                DFS_CODE_IS_MIN.push(newfrom, maxtoc+1, -1, elabel->first, tolabel->first);
                if(DFS_CODE[DFS_CODE_IS_MIN.size()-1] != DFS_CODE_IS_MIN[DFS_CODE_IS_MIN.size()-1]) return false;
                return project_is_min(tolabel->second);
            }
        }
        return true;
    }
    bool is_min(){
        // 此函数所有操作都基于新构建的子图上，与原数据集没有联系
        if(DFS_CODE.size()==1) return true;
        DFS_CODE.toGraph(GRAPH_IS_MIN); //这里重建的图重新给边编了id，且边没有用指针
        DFS_CODE_IS_MIN.clear();
        
        Projected_map3  root;
        EdgeList        edges;
        for(unsigned int from = 0;from<GRAPH_IS_MIN.size();from++){
            if(get_forward_root(GRAPH_IS_MIN, GRAPH_IS_MIN[from], edges)){  //在这个新的子图上：与顶点from有连接关系的所有边
                for(EdgeList::iterator it=edges.begin(); it!=edges.end(); it++){
                    root[GRAPH_IS_MIN[from].label][(*it)->elabel][GRAPH_IS_MIN[(*it)->to].label].push(0, *it, 0); //原图id都设为了0，这里没有现在图与原图的映射
                }
            }
        }
        
        Projected_iterator3 fromlabel = root.begin();
        Projected_iterator2 elabel    = fromlabel->second.begin();
        Projected_iterator1 tolabel   = elabel->second.begin();
        DFS_CODE_IS_MIN.push(0, 1, fromlabel->first, elabel->first, tolabel->first);    // 从每一条边开始查找？
        return (project_is_min(tolabel->second));
    }
    void print(Graph &g){
        // 输出当前图
        cout<<"t # "<<answer_number++<<endl;
        for(int i=0;i<g.vertex_size();i++){
            cout<<"v "<<i<<" "<<g[i].label<<endl;
        }
        for(int i=0;i<g.edge_size();i++){
            for(Vertex::edge_iterater it = g[i].edge.begin(); it!=g[i].edge.end(); it++){
                cout<<"e "<<(*it).from<<" "<<(*it).to<<" "<<(*it).elabel<<endl;
            }
        }
        cout<<endl;
    }
    void project(Projected &projected){ // 子图挖掘核心函数！ 递归调用，不断扩展当前子图
        // 检查当前子图是否频繁(两个节点一条边的，在外层并没有检测)
        unsigned int sup = support(projected);  //这个support只是计算一个一条边的频繁子图出现的频数（更大的子图需要PDFS结构体中的prev不断指向下一条边）
        
        if(sup < minsup) return ;
        // 下面进行min DFS检测，这个比minsup检测更耗时，所以后检测
        if(is_min() == false) return ;
        
        if(maxpat_max > maxpat_min && DFS_CODE.nodeCound()>maxpat_max) return ;
        
        // 满足两个检查，输出当前频繁集，DFS_CODE
//        if(DFS_CODE.nodeCound()==3) {
        Graph g;
        DFS_CODE.toGraph(g);
        print(g);
//        }
        
        // 扩展图 n -> (n+1)
        const RMPath &rmpath = DFS_CODE.buildRMPath();
        int minlabel = DFS_CODE[0].fromlabel;
        int maxtoc = DFS_CODE[rmpath[0]].to;
        
        Projected_map3 new_fwd_root;    //前向边
        Projected_map2 new_bck_root;    //后向边
        EdgeList edges;
        
        // 枚举所有扩展边的可能
        for(unsigned int n=0;n<projected.size(); n++){
            unsigned int id = projected[n].id;
            PDFS *cur = &projected[n];
            History history(TRANS[id], cur);
            
            // 后向边
            for(int i=(int)rmpath.size()-1; i>=1;i--){
                Edge *e = get_backward(TRANS[id], history[rmpath[i]], history[rmpath[0]], history);
                if(e){
                    new_bck_root[DFS_CODE[rmpath[i]].from][e->elabel].push(id, e, cur);
                }
            }
            // 最右节点扩展边
            if(get_forward_pure(TRANS[id], history[rmpath[0]], minlabel, history, edges))
                for(EdgeList::iterator it = edges.begin(); it!=edges.end(); it++)
                    new_fwd_root[maxtoc][(*it)->elabel][TRANS[id][(*it)->to].label].push(id, *it, cur);
            // 最右路径（除最右节点）扩展边
            for(int i=0;i<(int)rmpath.size();i++){
                if(get_forward_rmpath(TRANS[id], history[rmpath[i]], minlabel, history, edges)){
                    for(EdgeList::iterator it = edges.begin(); it!=edges.end(); it++){
                        new_fwd_root[DFS_CODE[rmpath[i]].from][(*it)->elabel][TRANS[id][(*it)->to].label].push(id, *it, cur);
                    }
                }
            }
        }
        
        // 测试所有扩展的子图：
        // backward
        for(Projected_iterator2 to = new_bck_root.begin(); to!=new_bck_root.end(); to++){
            for(Projected_iterator1 elabel = to->second.begin(); elabel!=to->second.end(); elabel++){
                DFS_CODE.push(maxtoc, to->first, -1, elabel->first, -1);
                project(elabel->second);
                DFS_CODE.pop();
            }
        }
        // forward
        for(Projected_riterator3 from = new_fwd_root.rbegin(); from!=new_fwd_root.rend(); from++){
            for(Projected_iterator2 elabel = from->second.begin(); elabel!=from->second.end(); elabel++){
                for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel!=elabel->second.end(); tolabel++){
                    DFS_CODE.push(from->first, maxtoc+1, -1, elabel->first, tolabel->first);
                    project(tolabel->second);
                    DFS_CODE.pop();
                }
            }
        }
        
        return ;
        
    }
    void run(){
        if(maxpat_min <= 1){    // one node subgraphs
            for(unsigned int id=0; id<TRANS.size(); id++){  // the graph id
                for(unsigned int nid=0; nid<TRANS[id].size(); nid++){   // vector<Vertex>
                    if(singleVertex[id][TRANS[id][nid].label] == 0) // the idth Graph
                        singleVertexLabel[TRANS[id][nid].label]+=1; // all Graph vertexLabel.
                    singleVertex[id][TRANS[id][nid].label] += 1;
                }
            }
            
            for(map<unsigned int, unsigned int>::iterator it = singleVertexLabel.begin(); it!=singleVertexLabel.end(); it++){
                if((*it).second < minsup) continue;
                unsigned int frequent_label = (*it).first;
                
                // the answer is stored in the format of graph.
                Graph g;
                g.resize(1);
                g[0].label = frequent_label;
                
                vector<unsigned int> counts(TRANS.size());
                for(map<unsigned int, map<unsigned int, unsigned int>>::iterator it2 = singleVertex.begin(); it2!=singleVertex.end(); it2++){
                    counts[(*it2).first] = (*it2).second[frequent_label];   // label在图(*it2).first中出现多少次
                }
                
            }
        }
        EdgeList edges;
        Projected_map3 root;
        for(unsigned int id=0; id<TRANS.size(); id++){
            Graph &g = TRANS[id];
            for(unsigned int from=0; from<g.size();from++){
                if(get_forward_root(g, g[from], edges)){ //对图id下顶点from有连接关系的所有边
                    for(EdgeList::iterator it = edges.begin(); it!=edges.end(); it++){  // 这里的edges每次在get_forward_root都会清空，只加入当前顶点的邻接边
                        root[g[from].label][(*it)->elabel][g[(*it)->to].label].push(id, *it, 0);    // 将每一条边，对应图id，PDFS编码为0，放入root中
                    }
                }
            }
        }
        for(Projected_iterator3 fromlabel=root.begin(); fromlabel!=root.end(); fromlabel++){
            for(Projected_iterator2 elabel = fromlabel->second.begin(); elabel!=fromlabel->second.end(); elabel++){
                for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel!=elabel->second.end(); tolabel++){
                    // 建立初始的两节点一条边的子图
                    DFS_CODE.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
                    
                    // 注意一个projected中保存的是一种类型边[from_label, elabel, to_label]在各个图中出现的情况：[图1 边1][图1 边3]...[图8 边5]...
                    // 扩展子图（增加边）的话，使用PDFS *prev指针来指向下一条边
                    
                    // 从每一个两节点一条边的子图进行搜索，递归树
                    project(tolabel->second);
                    DFS_CODE.pop();
                }
            }
        }
        
    }
    void run_time(){
        end = clock();
        cout<<"run time : "<<(double)(end-start)/CLOCKS_PER_SEC<<"s."<<endl;
    }
};

int main(int argc, const char * argv[]) {
    
    gSpan *g = new gSpan(argc, argv);
    g->run();
    g->run_time();
    
    return 0;
}





