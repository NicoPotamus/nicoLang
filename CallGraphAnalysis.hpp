#ifndef CALL_GRAPH_ANALYSIS_HPP
#define CALL_GRAPH_ANALYSIS_HPP

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>

// Call graph node for Tarjan's SCC algorithm
struct CallGraphNode {
    std::string functionName;
    std::vector<std::string> callees;  // functions this one calls
    int index = -1;        // Tarjan's index
    int lowlink = -1;      // Tarjan's lowlink
    bool onStack = false;  // Is it on the stack?
};

class CallGraphAnalyzer {
private:
    std::map<std::string, CallGraphNode> callGraph;
    std::vector<std::vector<std::string>> SCCs;  // Detected SCCs
    int tarjanIndex;
    std::vector<std::string> tarjanStack;

    void tarjanSCC(const std::string &funcName);

public:
    CallGraphAnalyzer() : tarjanIndex(0) {}

    // Record a function call: from -> to
    void recordCall(const std::string &caller, const std::string &callee);

    // Register a function (even if it doesn't call anything)
    void registerFunction(const std::string &funcName);

    // Run Tarjan's algorithm and detect SCCs
    void analyze();

    // Get detected SCCs
    const std::vector<std::vector<std::string>>& getSCCs() const { return SCCs; }

    // Print analysis results
    void printResults() const;

    // Clear all data
    void clear();
};

// Implementation (inline for header-only)
inline void CallGraphAnalyzer::recordCall(const std::string &caller, const std::string &callee) {
    callGraph[caller].functionName = caller;
    callGraph[caller].callees.push_back(callee);
}

inline void CallGraphAnalyzer::registerFunction(const std::string &funcName) {
    if (callGraph.find(funcName) == callGraph.end()) {
        callGraph[funcName].functionName = funcName;
    }
}

inline void CallGraphAnalyzer::tarjanSCC(const std::string &funcName) {
    CallGraphNode &node = callGraph[funcName];
    node.index = tarjanIndex;
    node.lowlink = tarjanIndex;
    tarjanIndex++;
    tarjanStack.push_back(funcName);
    node.onStack = true;

    // Visit all callees
    for (const auto &callee : node.callees) {
        if (callGraph.find(callee) == callGraph.end())
            continue;  // External or undefined function

        CallGraphNode &calleeNode = callGraph[callee];

        if (calleeNode.index == -1) {
            // Not yet visited
            tarjanSCC(callee);
            node.lowlink = std::min(node.lowlink, calleeNode.lowlink);
        } else if (calleeNode.onStack) {
            // In current SCC
            node.lowlink = std::min(node.lowlink, calleeNode.index);
        }
    }

    // Root of SCC?
    if (node.lowlink == node.index) {
        std::vector<std::string> scc;
        std::string w;
        do {
            w = tarjanStack.back();
            tarjanStack.pop_back();
            callGraph[w].onStack = false;
            scc.push_back(w);
        } while (w != funcName);

        // Check if it's a real SCC (has a cycle)
        if (scc.size() > 1 ||
            (scc.size() == 1 &&
             std::find(node.callees.begin(), node.callees.end(), funcName) != node.callees.end())) {
            SCCs.push_back(scc);
        }
    }
}

inline void CallGraphAnalyzer::analyze() {
    SCCs.clear();
    tarjanIndex = 0;
    tarjanStack.clear();

    // Reset all nodes
    for (auto &pair : callGraph) {
        pair.second.index = -1;
        pair.second.lowlink = -1;
        pair.second.onStack = false;
    }

    // Run Tarjan on all unvisited nodes
    for (auto &pair : callGraph) {
        if (pair.second.index == -1) {
            tarjanSCC(pair.first);
        }
    }
}

inline void CallGraphAnalyzer::printResults() const {
    if (!SCCs.empty()) {
        fprintf(stderr, "\n=== Strongly Connected Components (Recursive Functions) ===\n");
        for (const auto &scc : SCCs) {
            fprintf(stderr, "SCC: ");
            for (size_t i = 0; i < scc.size(); ++i) {
                fprintf(stderr, "%s", scc[i].c_str());
                if (i < scc.size() - 1) fprintf(stderr, " <-> ");
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "=========================================================\n\n");
    }
}

inline void CallGraphAnalyzer::clear() {
    callGraph.clear();
    SCCs.clear();
    tarjanIndex = 0;
    tarjanStack.clear();
}

#endif // CALL_GRAPH_ANALYSIS_HPP
