# Provider Decision Record

## Decision

The project uses The Guardian Open Platform for news metadata collection.

## Context

The original project considered NewsAPI. Early testing produced an HTTP 426 Upgrade Required response for the intended repeated data collection workflow.

For the final pipeline, Guardian was selected because it allowed a practical, auditable metadata retrieval workflow for the project scope.

## Consequences

Benefits:

- accessible provider for the project workflow
- time-stamped publication metadata
- reproducible query expressions
- suitable for a Guardian-only research prototype

Limitations:

- coverage is limited to Guardian retrieval
- keyword queries can be noisy
- relevant market-moving information from other outlets may be missing
- coverage varies by ticker

This provider choice should be described as a practical engineering constraint, not as a claim that Guardian coverage fully represents the UK financial news environment.
