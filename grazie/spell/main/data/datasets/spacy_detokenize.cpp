#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <math.h>
#include <map>
#include <queue>
#include <stack>
#include <unistd.h>
using namespace std;
#define ll long long
#define ld long double
#define sz(x) (int)(x).size()


//const int MAXN = 300001;
//
//int g[MAXN][29];

set<string> quote_suff = {"ve", "s ", "m ", "re", "t ", "d ", "ll", "VE", "S ", "M ", "RE", "T ", "D ", "LL"};
set<char> punct = {',', '.', ';', ':', '!', '?', '%'};

bool is_digit(char c){
    return int(c) >= int('0') and int(c) <= int('9');
}

string f(string s){
    string t = "";
    ll n = sz(s);
    ll x = 0, y = 0;
    for(int i = 0; i < n; i++){
        if(s[i] == '"'){
            x += 1;
        }
        if(s[i] == '\'' and !(i > 0 and i < n - 1 and s.substr(i - 1, 3) == "n't") and !(i < n - 4 and ((quote_suff.find(s.substr(i + 1, 2)) != quote_suff.end()) or (quote_suff.find(s.substr(i + 2, 2)) != quote_suff.end())))){
            y += 1;
        }
        if(s[i] == ' '){

            // don' t
            if(i > 0 and i + 2 < n and s[i - 1] == '\'' and quote_suff.find(s.substr(i + 1, 2)) != quote_suff.end()){
                continue;
            }

            // 4: 55 p.m.
            if(i > 0 and i + 2 < n and (s[i - 1] == ':' or s[i - 1] == '.') and is_digit(s[i + 1]) and is_digit(s[i + 2])){
                continue;
            }

            // n't
            if(i + 3 < n and s.substr(i + 1, 3) == "n't"){
                continue;
            }

            // 'm, 's, 've, 're
            if(i + 3 < n and s[i + 1] == '\'' and quote_suff.find(s.substr(i + 2, 2)) != quote_suff.end()){
                continue;
            }

            // punctuation
            if(i + 1 < n and punct.find(s[i + 1]) != punct.end()){
                continue;
            }

            // "
            if(i + 1 < n and s[i + 1] == '"'){
                if(x % 2 == 0){
                    t += s[i];
                }
                continue;
            }
            if(i > 0 and s[i - 1] == '"'){
                if(x % 2 == 0){
                    t += s[i];
                }
                continue;
            }

            // '
            if(i + 1 < n and s[i + 1] == '\''){
                if(y % 2 == 0){
                    t += s[i];
                }
                continue;
            }
            if(i > 0 and s[i - 1] == '\''){
                if(y % 2 == 0){
                    t += s[i];
                }
                continue;
            }

            // () [] {}
            if(i + 1 < n and (s[i + 1] == ')' or s[i + 1] == ']' or s[i + 1] == '}')){
                continue;
            }
            if(i > 0 and (s[i - 1] == '(' or s[i - 1] == '[' or s[i - 1] == '{')){
                continue;
            }

            t += s[i];
        }else{
            t += s[i];
        }
    }
    return t;
}

void solve(){
    while(cin) {
        string s;
        getline(cin, s);
        cout << f(s) << endl;
    };
}


int main(){
    freopen("/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/experiments/neuspell_bert/result_3.txt", "r", stdin);
    freopen("/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/experiments/neuspell_bert/result_4.txt", "w", stdout);
    ios::sync_with_stdio(0);cin.tie(0);
    int t = 1;
    while(t--)
        solve();
}
