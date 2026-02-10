"""Dummy OOLD resolver/backend for entity validation without a wiki."""

from oold.backend.interface import (
    Backend,
    ResolveParam,
    ResolveResult,
    SetBackendParam,
    SetResolverParam,
    StoreParam,
    StoreResult,
    set_backend,
    set_resolver,
)


class DummyBackend(Backend):
    """No-op resolver/backend for entity validation without a wiki."""

    def resolve_iris(self, iris):
        return {iri: None for iri in iris}

    def resolve(self, request: ResolveParam):
        return ResolveResult(nodes={iri: None for iri in request.iris})

    def store_jsonld_dicts(self, jsonld_dicts):
        return StoreResult(success=True)

    def store(self, request: StoreParam):
        return StoreResult(success=True)

    def query(self, query):
        return ResolveResult(nodes={})


def register():
    """Register dummy resolver/backend for all namespaces."""
    dummy = DummyBackend()
    for ns in ("Item", "Category", "Property", "File"):
        set_resolver(SetResolverParam(iri=ns, resolver=dummy))
        set_backend(SetBackendParam(iri=ns, backend=dummy))
