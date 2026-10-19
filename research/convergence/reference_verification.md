# Bibliographic verification

Checked 2026-10-15 against primary publisher, author preprint, or conference sources.
No existing PsiLogic paper was used to specify the algorithm or supply a proof.

| BibTeX key | Primary source checked | Relevant content and use |
|---|---|---|
| kingma2015adam | https://arxiv.org/abs/1412.6980 | Authors/title; original adaptive first/second moments. Original regret claim is not imported as a theorem here. |
| reddi2018convergence | https://openreview.net/pdf?id=ryQu7f-RZ | ICLR 2018 paper, explicit Adam failure and AMSGrad long-term memory. No claim that its exact example transfers unchanged with this epsilon/damping. |
| loshchilov2019decoupled | https://arxiv.org/abs/1711.05101 | Title/authors and ICLR 2019 venue; decoupled decay differs from L2 regularization under adaptive updates. |
| ghadimi2013stochastic | https://epubs.siam.org/doi/abs/10.1137/120880811 and https://arxiv.org/abs/1309.5549 | Authors, SIAM J. Optim. 23(4), 2341-2368 (2013); nonconvex stationarity complexity background. |
| chen2019adamtype | https://arxiv.org/abs/1808.02941 | Adam-type sufficient conditions and effective-step behavior; no direct theorem transfer asserted. |
| chen2023lion | https://arxiv.org/abs/2302.06675 | Title and twelve authors; sign-momentum algorithm background. Listed as preprint to avoid asserting unchecked proceedings metadata. |

The mathematical results in this package are proved in the manuscript rather than
inferred from abstracts. Sources establish context, attribution, and known obstacles.
