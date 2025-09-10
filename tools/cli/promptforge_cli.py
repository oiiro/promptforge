#!/usr/bin/env python3
"""
PromptForge CLI
Command-line interface for enterprise prompt management
"""

import click
import json
import yaml
from pathlib import Path
from typing import Optional
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import track
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from sdk.python.promptforge.client import PromptForgeClient, PromptForgeAdmin
from evaluation.deepeval_optimizer_minimal import HallucinationOptimizer, OptimizationConfig

console = Console()


@click.group()
@click.option('--team', '-t', default=None, help='Team name')
@click.option('--config', '-c', default='prompts', help='Config directory')
@click.option('--env', '-e', default='development', help='Environment')
@click.pass_context
def cli(ctx, team, config, env):
    """PromptForge Enterprise CLI - Manage prompts across teams"""
    ctx.ensure_object(dict)
    ctx.obj['team'] = team or os.getenv('PROMPTFORGE_TEAM', 'platform')
    ctx.obj['config'] = config
    ctx.obj['env'] = env


@cli.command()
@click.argument('name')
@click.option('--template', '-T', default='chain_of_thought', help='Template type')
@click.option('--description', '-d', required=True, help='Prompt description')
@click.option('--tags', '-g', multiple=True, help='Tags for categorization')
@click.pass_context
def create(ctx, name, template, description, tags):
    """Create a new prompt"""
    client = PromptForgeClient(
        team=ctx.obj['team'],
        config_path=ctx.obj['config'],
        environment=ctx.obj['env']
    )
    
    console.print(f"[bold blue]Creating prompt '{name}' for team '{ctx.obj['team']}'...[/bold blue]")
    
    try:
        prompt = client.create_prompt(
            name=name,
            template="",  # Will be loaded from template
            description=description,
            template_type=template,
            tags=list(tags)
        )
        
        console.print(f"[bold green]✅ Created prompt: {prompt.team}/{prompt.name}[/bold green]")
        console.print(f"   Path: {prompt.path}")
        console.print(f"   Version: {prompt.config['metadata']['version']}")
        console.print(f"   Status: {prompt.config['metadata']['status']}")
        
        # Show directory structure
        console.print("\n[bold]Directory structure created:[/bold]")
        for item in prompt.path.rglob("*"):
            if item.is_dir():
                level = len(item.relative_to(prompt.path).parts)
                console.print("  " * level + f"📁 {item.name}/")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument('name')
@click.option('--version', '-v', default=None, help='Specific version')
@click.option('--test-data', '-d', type=click.Path(exists=True), help='Test data file')
@click.option('--verbose', '-V', is_flag=True, help='Verbose output')
@click.pass_context
def test(ctx, name, version, test_data, verbose):
    """Test a prompt with evaluation metrics"""
    client = PromptForgeClient(
        team=ctx.obj['team'],
        config_path=ctx.obj['config'],
        environment=ctx.obj['env']
    )
    
    console.print(f"[bold blue]Testing prompt '{name}'...[/bold blue]")
    
    try:
        # Get the prompt
        prompt = client.get_prompt(name, version)
        
        # Load test data if provided
        test_data_dict = None
        if test_data:
            with open(test_data) as f:
                test_data_dict = json.load(f) if test_data.endswith('.json') else yaml.safe_load(f)
        
        # Run tests
        with console.status("[bold green]Running evaluation suite..."):
            results = client.test_prompt(prompt, test_data_dict)
        
        # Display results
        if results['passed']:
            console.print(f"[bold green]✅ All tests PASSED[/bold green]")
        else:
            console.print(f"[bold red]❌ Tests FAILED[/bold red]")
        
        # Create results table
        table = Table(title=f"Evaluation Results for {name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="yellow")
        table.add_column("Threshold", style="blue")
        table.add_column("Status", style="green")
        
        for metric, score in results['results'].items():
            threshold = results['thresholds'].get(f"{metric}_threshold", 0.0)
            status = "✅" if score >= threshold else "❌"
            table.add_row(
                metric,
                f"{score:.3f}",
                f"{threshold:.3f}",
                status
            )
        
        console.print(table)
        
        if verbose:
            console.print("\n[bold]Detailed Results:[/bold]")
            console.print(json.dumps(results, indent=2))
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument('name')
@click.option('--iterations', '-i', default=5, help='Number of optimization iterations')
@click.option('--target-hallucination', '-h', default=0.95, help='Target hallucination score')
@click.option('--test-data', '-d', type=click.Path(exists=True), help='Test data file')
@click.pass_context
def optimize(ctx, name, iterations, target_hallucination, test_data):
    """Optimize a prompt for low hallucination using Chain-of-Thought"""
    client = PromptForgeClient(
        team=ctx.obj['team'],
        config_path=ctx.obj['config'],
        environment=ctx.obj['env']
    )
    
    console.print(f"[bold blue]Optimizing prompt '{name}'...[/bold blue]")
    
    try:
        # Get the prompt
        prompt = client.get_prompt(name)
        
        # Configure optimizer
        config = OptimizationConfig(
            max_iterations=iterations,
            target_hallucination_score=target_hallucination,
            enable_cot=True,
            cot_style="structured"
        )
        
        optimizer = HallucinationOptimizer(config)
        
        # Load test data
        test_cases = []
        if test_data:
            with open(test_data) as f:
                test_cases = json.load(f) if test_data.endswith('.json') else yaml.safe_load(f)
        
        # Run optimization
        console.print(f"Target hallucination score: {target_hallucination:.0%}")
        
        with console.status("[bold green]Running optimization..."):
            results = optimizer.optimize_prompt(
                base_prompt=prompt.config['prompt']['template'],
                test_cases=test_cases
            )
        
        # Display results
        console.print(f"[bold green]✅ Optimization complete![/bold green]")
        console.print(f"   Iterations: {results['iterations']}")
        console.print(f"   Improvement: {results['improvement']:.3f}")
        
        # Show score evolution
        table = Table(title="Score Evolution")
        table.add_column("Iteration", style="cyan")
        table.add_column("Hallucination", style="yellow")
        table.add_column("Faithfulness", style="blue")
        table.add_column("Relevancy", style="green")
        
        for h in results.get('history', []):
            table.add_row(
                str(h['iteration']),
                f"{h['scores'].get('hallucination', 0):.3f}",
                f"{h['scores'].get('faithfulness', 0):.3f}",
                f"{h['scores'].get('relevancy', 0):.3f}"
            )
        
        console.print(table)
        
        # Save optimized prompt
        save = click.confirm("Save optimized prompt?")
        if save:
            # Update prompt with optimized template
            prompt.config['prompt']['template'] = results['optimized_prompt']
            prompt.config['prompt']['parameters'].update(results['configuration'])
            prompt.save()
            console.print(f"[bold green]✅ Saved optimized prompt[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument('name')
@click.option('--environment', '-e', required=True, type=click.Choice(['staging', 'production']))
@click.option('--no-approval', is_flag=True, help='Skip approval requirement')
@click.pass_context
def deploy(ctx, name, environment, no_approval):
    """Deploy a prompt to staging or production"""
    client = PromptForgeClient(
        team=ctx.obj['team'],
        config_path=ctx.obj['config'],
        environment=ctx.obj['env']
    )
    
    console.print(f"[bold blue]Deploying '{name}' to {environment}...[/bold blue]")
    
    try:
        prompt = client.get_prompt(name)
        
        # Confirm deployment
        if environment == 'production':
            if not click.confirm(f"Deploy {name} to PRODUCTION?"):
                console.print("[yellow]Deployment cancelled[/yellow]")
                return
        
        # Deploy
        result = client.deploy_prompt(
            prompt=prompt,
            environment=environment,
            require_approval=not no_approval
        )
        
        if result.get('status') == 'approval_required':
            console.print(f"[yellow]⏳ {result['message']}[/yellow]")
            console.print(f"   Request ID: {result['request']['timestamp']}")
        else:
            console.print(f"[bold green]✅ Deployed to {environment}[/bold green]")
            console.print(f"   Version: {result['version']}")
            console.print(f"   Timestamp: {result['timestamp']}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option('--team', '-t', default=None, help='Filter by team')
@click.option('--status', '-s', default=None, help='Filter by status')
@click.option('--tags', '-g', multiple=True, help='Filter by tags')
@click.pass_context
def list(ctx, team, status, tags):
    """List available prompts"""
    client = PromptForgeClient(
        team=ctx.obj['team'],
        config_path=ctx.obj['config'],
        environment=ctx.obj['env']
    )
    
    prompts = client.list_prompts(
        team=team,
        status=status,
        tags=list(tags) if tags else None
    )
    
    if not prompts:
        console.print("[yellow]No prompts found[/yellow]")
        return
    
    # Create table
    table = Table(title="Available Prompts")
    table.add_column("Name", style="cyan")
    table.add_column("Team", style="yellow")
    table.add_column("Version", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Score", style="magenta")
    
    for prompt in prompts:
        hallucination = prompt.get('metrics', {}).get('hallucination_score', 0)
        score_str = f"{hallucination:.2f}" if hallucination > 0 else "N/A"
        
        table.add_row(
            prompt['name'],
            prompt['team'],
            prompt['current_version'],
            prompt['status'],
            score_str
        )
    
    console.print(table)


@cli.command()
@click.argument('name')
@click.argument('version1')
@click.argument('version2')
@click.option('--test-data', '-d', type=click.Path(exists=True), help='Test data file')
@click.pass_context
def compare(ctx, name, version1, version2, test_data):
    """Compare two versions of a prompt"""
    client = PromptForgeClient(
        team=ctx.obj['team'],
        config_path=ctx.obj['config'],
        environment=ctx.obj['env']
    )
    
    console.print(f"[bold blue]Comparing {name} v{version1} vs v{version2}...[/bold blue]")
    
    try:
        # Load test data if provided
        test_data_dict = None
        if test_data:
            with open(test_data) as f:
                test_data_dict = json.load(f) if test_data.endswith('.json') else yaml.safe_load(f)
        
        # Compare versions
        comparison = client.compare_versions(
            prompt_name=name,
            version1=version1,
            version2=version2,
            test_data=test_data_dict
        )
        
        # Display comparison
        table = Table(title=f"Version Comparison: {name}")
        table.add_column("Metric", style="cyan")
        table.add_column(f"v{version1}", style="yellow")
        table.add_column(f"v{version2}", style="blue")
        table.add_column("Change", style="green")
        
        # Get all metrics
        all_metrics = set()
        all_metrics.update(comparison['version1']['results'].keys())
        all_metrics.update(comparison['version2']['results'].keys())
        
        for metric in sorted(all_metrics):
            v1_score = comparison['version1']['results'].get(metric, 0)
            v2_score = comparison['version2']['results'].get(metric, 0)
            change = v2_score - v1_score
            
            change_str = f"{change:+.3f}"
            if change > 0:
                change_str = f"[green]{change_str}[/green]"
            elif change < 0:
                change_str = f"[red]{change_str}[/red]"
            
            table.add_row(
                metric,
                f"{v1_score:.3f}",
                f"{v2_score:.3f}",
                change_str
            )
        
        console.print(table)
        
        # Summary
        if comparison['improvements']:
            console.print("\n[bold green]Improvements:[/bold green]")
            for metric, value in comparison['improvements'].items():
                console.print(f"  • {metric}: +{value:.3f}")
        
        if comparison['regressions']:
            console.print("\n[bold red]Regressions:[/bold red]")
            for metric, value in comparison['regressions'].items():
                console.print(f"  • {metric}: -{value:.3f}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show team and environment status"""
    console.print(f"[bold]PromptForge Status[/bold]")
    console.print(f"  Team: {ctx.obj['team']}")
    console.print(f"  Environment: {ctx.obj['env']}")
    console.print(f"  Config Path: {ctx.obj['config']}")
    
    # Try to load team configuration
    try:
        client = PromptForgeClient(
            team=ctx.obj['team'],
            config_path=ctx.obj['config'],
            environment=ctx.obj['env']
        )
        
        team_config = client.team_manager.get_team_config(ctx.obj['team'])
        
        console.print(f"\n[bold]Team Configuration:[/bold]")
        console.print(f"  Name: {team_config.get('name', 'N/A')}")
        console.print(f"  Lead: {team_config.get('lead', 'N/A')}")
        console.print(f"  Members: {len(team_config.get('members', []))}")
        
        console.print(f"\n[bold]Permissions:[/bold]")
        for perm in team_config.get('permissions', []):
            console.print(f"  • {perm}")
        
        console.print(f"\n[bold]Evaluation Overrides:[/bold]")
        overrides = team_config.get('evaluation_overrides', {})
        for key, value in overrides.items():
            console.print(f"  • {key}: {value}")
        
    except Exception as e:
        console.print(f"[yellow]Unable to load team configuration: {e}[/yellow]")


if __name__ == '__main__':
    cli()