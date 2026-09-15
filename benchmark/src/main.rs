#![feature(f16)]

mod benchmark;
mod cli;
mod report;

use clap::Parser;

use crate::cli::Cli;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = Cli::parse().into_plan()?;
    let records = benchmark::run(&plan)?;

    report::print_results_table(&records);
    report::write_records(plan.csv_output, plan.json_output, &records)?;
    Ok(())
}
