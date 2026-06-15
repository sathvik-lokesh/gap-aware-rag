# Frostpath Deployment & Maintenance Notes

Aurelia Frostpath fleets are deployed at customer cold-storage sites under a
robots-as-a-service contract. A typical site runs between 8 and 40 units
coordinated by the Aurelia FleetMind scheduler.

Maintenance: heated treads should be inspected every 500 operating hours for
wear. The thermal jacket on the battery should be replaced annually. Anti-fog
lens covers are consumable and rated for roughly six months of continuous cold
operation.

Charging: units return to heated docking stations. A full charge from 10% takes
about 75 minutes. Aurelia recommends keeping at least 20% of a fleet docked and
warm at all times to absorb demand spikes.

Safety: Frostpath robots halt if a human is detected within 1.2 meters and
resume after the path clears. All units log telemetry to the FleetMind dashboard
for predictive-maintenance alerts.
