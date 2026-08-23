# Governance

Hexaform uses a maintainer-led, review-based model. Repository maintainers own
release scope, security response, and the scientific validity gates. Changes to
core topology, element orientation, quality thresholds, or Exodus semantics
require review from a maintainer familiar with that subsystem.

The protected development path is feature branch → pull request → CI → review →
`master`. Direct pushes to `master` should be reserved for repository bootstrap
or urgent recovery. Releases are tagged from a clean `master`, accompanied by a
generated technical report and changelog update.

Roadmap status is evidence-based: “complete” means implemented, tested, and
documented. Experimental paths remain clearly labeled and are never promoted by
silencing validation failures.
