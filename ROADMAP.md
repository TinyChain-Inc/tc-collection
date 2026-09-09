# tc-collection roadmap

This file contains only unimplemented collection work. Current behavior is
specified by the crate README and transactional collection contract.

## Persistent Tensor backend

- Replace the private in-memory Tensor backend without changing the public
  Tensor State variant, codec, route family, or lifecycle.
- Preserve bounded streaming and device admission through the existing owner
  contracts.
- Complete transaction visibility, conflict, finalization, restart
  materialization, and cancellation tests before removing the old backend.

## Parity verification

- Finish any open BTree and Table cases in the transactional contract.
- Apply the same contract to persistent Tensor before declaring the migration
  complete.
