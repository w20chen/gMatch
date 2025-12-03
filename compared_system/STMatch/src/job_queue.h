#pragma once

#include "graph.h"
#include "config.h"

namespace STMatch {

  typedef struct {
    graph_node_t nodes[PAT_SIZE];
  } Job;

  typedef struct {
    Job* q;
    int start_level;
    graph_node_t length;
    graph_node_t cur;
    int mutex = 0;
  } JobQueue;

  struct JobQueuePreprocessor {

    JobQueue q;


    JobQueuePreprocessor(Graph& g, PatternPreprocessor& p) {
      std::cout << "Constructing JobQueuePreprocessor" << std::endl;
      TIME_INIT();
      TIME_START();
      std::vector<graph_node_t> vr, vc;
      for (graph_node_t r = 0; r < g.nnodes; r++) {
        for (ull j = g.rowptr[r]; j < g.rowptr[r + 1]; j++) {
          graph_node_t c = g.colidx[j];
          if ((!LABELED && p.partial[0][0] == 1 && r > c) || LABELED || p.partial[0][0] != 1) {
            if ((g.vertex_label[r] == (1 << p.vertex_labels[0])) && (g.vertex_label[c] == (1 << p.vertex_labels[1]))) {
              if (g.rowptr[r + 1] - g.rowptr[r] >= p.pat.degree[0] && g.rowptr[c + 1] - g.rowptr[c] >= p.pat.degree[1]) {
                bool valid = false;
                for (ull d = g.rowptr[c]; d < g.rowptr[c + 1]; d++) {
                  graph_node_t v = g.colidx[d];
                  if (g.rowptr[v + 1] - g.rowptr[v] >= p.pat.degree[2]) {
                    valid = true;
                    break;
                  }
                }

                if (valid) {
                  vr.push_back(r);
                  vc.push_back(c);
                }
              }
            }
          }
        }
      }
      std::cout << "vr.size() " << vr.size() << std::endl;
      q.q = new Job[vr.size()];
      for (ull i = 0; i < vr.size(); i++) {
        (q.q[i].nodes)[0] = vr[i];
        (q.q[i].nodes)[1] = vc[i];
      }
      q.length = vr.size();
      q.cur = 0;
      q.start_level = 2;
      TIME_END();
      PRINT_TOTAL_TIME("STMatch filtering on the CPU");
    }

    JobQueue* to_gpu() {
      JobQueue qcopy = q;
      printf("attempt to allocate %lf MB\n", sizeof(Job) * q.length / 1024. / 1024.);
      cudaMalloc(&qcopy.q, sizeof(Job) * q.length);
      cudaMemcpy(qcopy.q, q.q, sizeof(Job) * q.length, cudaMemcpyHostToDevice);

      JobQueue* gpu_q;
      cudaMalloc(&gpu_q, sizeof(JobQueue));
      cudaMemcpy(gpu_q, &qcopy, sizeof(JobQueue), cudaMemcpyHostToDevice);
      return gpu_q;
    }
  };
}